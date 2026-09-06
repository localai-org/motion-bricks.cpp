# Human led design document

Don't modify this unless instructed.

# Questions to loop back to

- where is the C API documentation
- Are low-level functions exposed so things like styles can be handle externally? could someone just use it for inference?
- the camera jerks or shakes when we reach a corner or a transition
- What is the difference between our context and upstreams?
- Gumbel sampling

# End goal

We want to use motion-bricks.cpp with kimodo.cpp animation key-frames and we want to be able to use these in a web demo.

So we have a web demo that takes some key-frames for particular movements and styles of movements, like in the upstream demo, including the original key-frames. Like in the upstream demo we are going to have a bunch of movements we can do with the WASD keys and jump or some other user definable motion.

Eventually, we want to have have a configurable list of movements, that include a keyframe, an action (e.g. move forward at 1m/s, move up (jump)), so we could define a walking action with a keyframe from someone walking and an action of moving forwards, and we could define a forward roll action by specifying one or more keyframes from rolling forwards and also a moving forward action (but perhaps faster forward motion).

The motion-bricks.cpp library should expose a C API for FFI which is compatible with libraries like purego. So purego can't predict struct layouts, so struct based interfaces should have a constructor for allocating structs and getters and setters.

The demo server should be written in Go and use purego (or similar). The frontend should use plain HTML/CSS and three.js.

# Bonus Goal

Use motion-bricks.cpp with a physically simulated robot in the C++ backend using the Mujoco headless C API and GEAR-SONIC and render the results in the browser.

# Process

- First we want to produce exemplar outputs from upstream
- We want to create parity tests for each component, including every layer of the neural network.
- We want to ensure that components feeding and consuming input and output from the neural network also match upstream in terms of functionality
- We need to ensure we are not using settings in our implementation by default which are not set as default in upstream
- We should use GGML with Vulkan and CPU backends, first CPU and then Vulkan parity should be achieved
- We can use ../kimodo.cpp and ../skin-tokens.cpp/ as a reference, you should scan these projects before designing and check them when trying to solve an issue for an existing solution
- We are on NixOS and can use a Nix flake to get depenencies for GGML/C++ and Vulkan, the machine we are on has a functioning NVIDIA Vulkan GPU and an AMD one. We want to use the NVIDIA gfx card.
- For the upstream depencies, if we can use Nix that is great, but otherwise we can use Docker
- Don't rely on the current machine's paths, use distro agnostic paths and env vars. parameterise anything that we need to specify in the CMake file or a .env
- We need to fuzz the C++ components except for GGUF input (we trust that GGML has that covered). We should do that with the ASan and UBSAN enabled.
- We should use GGML as git submodule and if we need to patch it for any reason we can apply patches during the project configuration phase
- Use headless chrome (from nix) to perform QA after changes. Use a few different actions and take screenshots before during and after
