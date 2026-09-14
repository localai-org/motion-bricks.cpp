package motionbricks

import (
	"math"
	"os"
	"testing"
	"unsafe"

	"github.com/ebitengine/purego"
)

const (
	statusOK      = uint32(0)
	deviceAuto    = uint32(0)
	deviceVulkan  = uint32(2)
	abiVersion    = uint32(1)
	errorCapacity = uint64(256)
)

// TestOpaqueABIRoundTrip intentionally mirrors no C struct. Every object is an
// opaque uintptr and every record field is accessed through a function.
func TestOpaqueABIRoundTrip(t *testing.T) {
	library := os.Getenv("MOTIONBRICKS_LIB")
	if library == "" {
		t.Skip("set MOTIONBRICKS_LIB to the motion-bricks.cpp shared library")
	}

	handle, err := openLibrary(library)
	if err != nil {
		t.Fatalf("dlopen %s: %v", library, err)
	}
	defer closeLibrary(handle)

	var version func() uint32
	var optionsCreate func(unsafe.Pointer, unsafe.Pointer, uint64) uint32
	var optionsFree func(uintptr)
	var optionsSetDevice func(uintptr, uint32, unsafe.Pointer, uint64) uint32
	var optionsGetDevice func(uintptr, unsafe.Pointer, unsafe.Pointer, uint64) uint32
	var commandCreate func(unsafe.Pointer, unsafe.Pointer, uint64) uint32
	var commandFree func(uintptr)
	var commandSetMovement func(uintptr, float32, float32, float32, unsafe.Pointer, uint64) uint32
	var commandGetMovement func(uintptr, unsafe.Pointer, unsafe.Pointer, unsafe.Pointer, unsafe.Pointer, uint64) uint32

	purego.RegisterLibFunc(&version, handle, "mb_abi_version")
	purego.RegisterLibFunc(&optionsCreate, handle, "mb_runtime_options_create")
	purego.RegisterLibFunc(&optionsFree, handle, "mb_runtime_options_free")
	purego.RegisterLibFunc(&optionsSetDevice, handle, "mb_runtime_options_set_device")
	purego.RegisterLibFunc(&optionsGetDevice, handle, "mb_runtime_options_get_device")
	purego.RegisterLibFunc(&commandCreate, handle, "mb_command_create")
	purego.RegisterLibFunc(&commandFree, handle, "mb_command_free")
	purego.RegisterLibFunc(&commandSetMovement, handle, "mb_command_set_movement_direction")
	purego.RegisterLibFunc(&commandGetMovement, handle, "mb_command_get_movement_direction")

	if got := version(); got != abiVersion {
		t.Fatalf("ABI version = %d, want %d", got, abiVersion)
	}

	errorBuffer := make([]byte, errorCapacity)
	errorPointer := unsafe.Pointer(&errorBuffer[0])
	var options uintptr
	if status := optionsCreate(unsafe.Pointer(&options), errorPointer, errorCapacity); status != statusOK {
		t.Fatalf("options create status %d: %q", status, errorBuffer)
	}
	if options == 0 {
		t.Fatal("options create returned a null handle")
	}
	defer optionsFree(options)

	var device uint32 = math.MaxUint32
	if status := optionsGetDevice(options, unsafe.Pointer(&device), errorPointer, errorCapacity); status != statusOK {
		t.Fatalf("get default device status %d", status)
	}
	if device != deviceAuto {
		t.Fatalf("default device = %d, want auto", device)
	}
	if status := optionsSetDevice(options, deviceVulkan, errorPointer, errorCapacity); status != statusOK {
		t.Fatalf("set Vulkan status %d", status)
	}
	if status := optionsGetDevice(options, unsafe.Pointer(&device), errorPointer, errorCapacity); status != statusOK || device != deviceVulkan {
		t.Fatalf("Vulkan round trip status=%d device=%d", status, device)
	}

	var command uintptr
	if status := commandCreate(unsafe.Pointer(&command), errorPointer, errorCapacity); status != statusOK {
		t.Fatalf("command create status %d", status)
	}
	if command == 0 {
		t.Fatal("command create returned a null handle")
	}
	defer commandFree(command)

	if status := commandSetMovement(command, 1.0, 0.0, -0.25, errorPointer, errorCapacity); status != statusOK {
		t.Fatalf("set movement status %d", status)
	}
	var x, y, z float32
	if status := commandGetMovement(command, unsafe.Pointer(&x), unsafe.Pointer(&y), unsafe.Pointer(&z), errorPointer, errorCapacity); status != statusOK {
		t.Fatalf("get movement status %d", status)
	}
	if x != 1.0 || y != 0.0 || z != -0.25 {
		t.Fatalf("movement = (%g, %g, %g)", x, y, z)
	}
}
