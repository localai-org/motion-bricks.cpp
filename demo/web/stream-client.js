// Timestamped, bounded playback. Rendering never drives the server clock.
export class PoseBuffer {
  constructor(delay=.15) { this.delay=delay;this.reset(); }
  reset() {this.frames=[];this.epoch=null;this.time=null;this.waiting=true;this.underruns=0;this.received=0;}
  push(frame) {
    const valid=(a,n)=>Array.isArray(a)&&a.length===n&&a.every(x=>Number.isFinite(x)&&Math.abs(x)<=10000);
    if(frame.type!=='frame'||!Number.isSafeInteger(frame.epoch)||!Number.isSafeInteger(frame.tick)||!Number.isFinite(frame.time)||frame.time<0||!valid(frame.root,3)||!valid(frame.rotations,136)||(frame.physics&&!valid(frame.physical,90)))throw new Error('Invalid or extreme streamed pose');
    for(let i=0;i<136;i+=4){let norm=0;for(let j=0;j<4;j++)norm+=frame.rotations[i+j]**2;if(Math.abs(norm-1)>.02)throw new Error('Invalid streamed quaternion');}
    if(frame.collision_transforms!==undefined){
      const c=frame.collision_transforms;
      if(!Array.isArray(c)||c.length%7||c.length>1024*7||!valid(c,c.length))throw new Error('Invalid collision transforms');
      for(let i=0;i<c.length;i+=7){const norm=c.slice(i+3,i+7).reduce((sum,x)=>sum+x*x,0);if(Math.abs(norm-1)>.02)throw new Error('Invalid collision quaternion');}
    }
    if(this.epoch!==null&&frame.epoch<this.epoch)return;
    if(frame.epoch!==this.epoch){this.reset();this.epoch=frame.epoch;}
    const last=this.frames.at(-1);
    if(last&&frame.time<=last.time){if(frame.time===last.time)this.frames[this.frames.length-1]=frame;return;}
    this.frames.push(frame);this.received++;
    if(this.frames.length>128){this.frames.splice(0,this.frames.length-128);this.waiting=true;this.time=null;}
  }
  sample(delta) {
    const frames=this.frames;if(!frames.length)return null;
    const latest=frames.at(-1);
    if(this.time===null){
      if(latest.time-frames[0].time<this.delay&&!latest.paused)return {a:frames[0],b:frames[0],alpha:0,buffering:true};
      this.time=frames[0].time;this.waiting=false;
    }
    if(this.waiting&&latest.time-this.time>=this.delay*.75)this.waiting=false;
    if(!this.waiting){
      const lead=latest.time-this.time;
      const rate=1+Math.max(-.02,Math.min(.02,(lead-this.delay)*.1));
      const next=this.time+Math.max(0,Math.min(delta,.1))*rate;
      if(next>latest.time&&!latest.paused){this.time=latest.time;this.waiting=true;this.underruns++;}
      else this.time=Math.min(latest.time,next);
    }
    while(frames.length>2&&frames[1].time<=this.time)frames.shift();
    const a=frames[0],b=frames[1]??a;
    const alpha=b.time>a.time?Math.max(0,Math.min(1,(this.time-a.time)/(b.time-a.time))):0;
    return {a,b,alpha,buffering:this.waiting};
  }
}

export class StreamClient {
  constructor(callback) {this.callback=callback;this.buffer=new PoseBuffer();this.seq=0;this.owner=false;this.closed=false;this.retry=null;this.connect();this.ping=setInterval(()=>this.send('ping'),10000);}
  connect(){
    if(this.closed)return;
    const token=sessionStorage.getItem('motionbricks-stream-token')??'';
    const url=new URL('/api/stream/socket',location.href);url.protocol=location.protocol==='https:'?'wss:':'ws:';if(token)url.searchParams.set('resume',token);
    const socket=this.socket=new WebSocket(url);
    socket.onmessage=event=>{try{
      const message=JSON.parse(event.data);
      if(message.type==='hello'){
        if(message.version!==1)throw new Error('Unsupported stream version');
        this.owner=message.owner;this.seq=Math.max(this.seq,message.last_seq??0);if(message.token)sessionStorage.setItem('motionbricks-stream-token',message.token);
        this.callback(message);if(this.owner)this.send('resume');return;
      }
      if(message.type==='frame')this.buffer.push(message);
      this.callback(message);
    }catch(error){this.callback({type:'error',error:error.message});socket.close();}};
    socket.onclose=()=>{this.owner=false;this.callback({type:'disconnected'});if(!this.closed)this.retry=setTimeout(()=>this.connect(),750);};
    socket.onerror=()=>this.callback({type:'connection_error'});
  }
  send(type,values={}){if(this.socket?.readyState!==WebSocket.OPEN||(!this.owner&&type!=='ping'))return false;if(this.socket.bufferedAmount>16384){this.callback({type:'error',error:'Command connection is congested'});return false;}this.socket.send(JSON.stringify({type,seq:++this.seq,...values}));return true;}
  close(){this.closed=true;clearInterval(this.ping);clearTimeout(this.retry);this.socket?.close();}
}
