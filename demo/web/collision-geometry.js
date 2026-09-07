import * as THREE from './vendor/three.module.min.js';

// Geometry stays in the compiled MuJoCo geom-local basis. Streamed rotations
// already include the world-basis change to the viewer; do not apply it twice.
export class CollisionGeometry {
  constructor(shapes) {
    if(!Array.isArray(shapes)||shapes.length>1024)throw new Error('Invalid collision definitions');
    this.group=new THREE.Group();this.group.visible=false;this.meshes=[];
    // Diagnostic surfaces do not need lighting/PBR or two transparent passes.
    // This also keeps software-rendered headless clients responsive.
    this.material=new THREE.MeshBasicMaterial({color:0x80b9d4,transparent:true,opacity:.6,depthWrite:false,side:THREE.FrontSide,polygonOffset:true,polygonOffsetFactor:1,polygonOffsetUnits:1});
    this.outline=new THREE.LineBasicMaterial({color:0xb0d9e8,transparent:true,opacity:.5,depthWrite:false});
    this.other=new THREE.Quaternion();
    try {
      for(const shape of shapes) {
        if(!Array.isArray(shape.size)||shape.size.length!==3||!shape.size.every(x=>Number.isFinite(x)&&x>=0&&x<10000))throw new Error('Invalid collision size');
        const [x,y,z]=shape.size;let geometry;
        switch(shape.type) {
          case 2:geometry=new THREE.SphereGeometry(x,20,12);break;
          case 3:geometry=new THREE.CapsuleGeometry(x,2*y,6,20);geometry.rotateX(Math.PI/2);break;
          case 4:geometry=new THREE.SphereGeometry(1,20,12);geometry.scale(x,y,z);break;
          case 5:geometry=new THREE.CylinderGeometry(x,x,2*y,24);geometry.rotateX(Math.PI/2);break;
          case 6:geometry=new THREE.BoxGeometry(2*x,2*y,2*z);break;
          case 7:
            if(!Array.isArray(shape.vertices)||shape.vertices.length===0||shape.vertices.length%3||shape.vertices.length>1000000||!shape.vertices.every(v=>Number.isFinite(v)&&Math.abs(v)<10000))throw new Error('Invalid collision hull');
            if(!Array.isArray(shape.indices)||shape.indices.length%3||shape.indices.length>1000000||!shape.indices.every(i=>Number.isInteger(i)&&i>=0&&i<shape.vertices.length/3))throw new Error('Invalid collision hull indices');
            geometry=new THREE.BufferGeometry();geometry.setAttribute('position',new THREE.Float32BufferAttribute(shape.vertices,3));geometry.setIndex(shape.indices);break;
          default:throw new Error(`Unsupported collision shape ${shape.type}`);
        }
        const mesh=new THREE.Mesh(geometry,this.material);mesh.name=shape.name;mesh.renderOrder=3;
        const edges=new THREE.LineSegments(new THREE.EdgesGeometry(geometry,25),this.outline);edges.renderOrder=4;mesh.add(edges);
        this.group.add(mesh);this.meshes.push(mesh);
      }
      // Keep exact hulls; batch their rigid transforms on the GPU instead of
      // issuing two draw calls for each geom. Larger future scenes fall back
      // to ordinary meshes rather than exceeding the uniform matrix budget.
      if(this.meshes.length>0&&this.meshes.length<=48)this.batch();
    }catch(error){this.dispose();throw error;}
  }
  batch() {
    const matrices=this.meshes.map(mesh=>mesh.matrix);
    const vertexShader=`attribute float shapeIndex;
      uniform mat4 poses[${this.meshes.length}];
      void main(){gl_Position=projectionMatrix*modelViewMatrix*poses[int(shapeIndex+0.5)]*vec4(position,1.0);}`;
    const fragmentShader=`uniform vec3 tint;uniform float opacity;
      void main(){gl_FragColor=vec4(tint,opacity);
        #include <tonemapping_fragment>
        #include <colorspace_fragment>
      }`;
    const makeMaterial=(color,opacity)=>new THREE.ShaderMaterial({
      uniforms:{poses:{value:matrices},tint:{value:new THREE.Color(color)},opacity:{value:opacity}},
      vertexShader,fragmentShader,transparent:true,depthWrite:false,side:THREE.FrontSide,
    });
    this.batchMaterials=[makeMaterial(this.material.color,this.material.opacity),makeMaterial(this.outline.color,this.outline.opacity)];
    for(let pass=0;pass<2;pass++) {
      const positions=[],ids=[],indices=[];
      for(let i=0;i<this.meshes.length;i++) {
        const source=pass===0?this.meshes[i].geometry:this.meshes[i].children[0].geometry;
        const attribute=source.attributes.position,base=positions.length/3;
        for(let j=0;j<attribute.count;j++){positions.push(attribute.getX(j),attribute.getY(j),attribute.getZ(j));ids.push(i);}
        if(source.index)for(let j=0;j<source.index.count;j++)indices.push(base+source.index.getX(j));
        else for(let j=0;j<attribute.count;j++)indices.push(base+j);
      }
      const geometry=new THREE.BufferGeometry();
      geometry.setAttribute('position',new THREE.Float32BufferAttribute(positions,3));
      geometry.setAttribute('shapeIndex',new THREE.Float32BufferAttribute(ids,1));geometry.setIndex(indices);
      const object=pass===0?new THREE.Mesh(geometry,this.batchMaterials[pass]):new THREE.LineSegments(geometry,this.batchMaterials[pass]);
      object.frustumCulled=false;object.renderOrder=3+pass;this.group.add(object);
    }
    for(const mesh of this.meshes)mesh.visible=false;
  }
  pose(a,b,alpha) {
    if(!a||!b||a.length!==this.meshes.length*7||b.length!==a.length){this.group.visible=false;return;}
    for(let i=0;i<this.meshes.length;i++) {
      const mesh=this.meshes[i],at=i*7;
      mesh.position.set(a[at]+alpha*(b[at]-a[at]),a[at+1]+alpha*(b[at+1]-a[at+1]),a[at+2]+alpha*(b[at+2]-a[at+2]));
      mesh.quaternion.fromArray(a,at+3).slerp(this.other.fromArray(b,at+3),alpha);
      mesh.updateMatrix();
    }
    this.group.visible=true;
  }
  dispose() {
    this.group.removeFromParent();
    this.group.traverse(object=>object.geometry?.dispose());
    this.group.clear();this.material.dispose();this.outline.dispose();
    this.batchMaterials?.forEach(material=>material.dispose());
  }
}
