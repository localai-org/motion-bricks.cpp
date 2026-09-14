//go:build windows

package motionbricks

import (
	"path/filepath"
	"runtime"
	"syscall"
	"unsafe"
)

// PureGo supports windows/amd64, but its dlfcn wrappers are POSIX-only, so the
// loader uses the Win32 module API directly. RegisterLibFunc and RegisterFunc
// accept the resulting handles and addresses unchanged.

// loadWithAlteredSearchPath resolves a module's own dependencies from the
// directory holding it rather than the directory of the running executable,
// which is how the shared library finds the GGML backend modules beside it.
// The standard library exposes LoadLibraryW but not LoadLibraryExW.
const loadWithAlteredSearchPath = 0x00000008

var procLoadLibraryEx = syscall.NewLazyDLL("kernel32.dll").NewProc("LoadLibraryExW")

// openLibrary loads a shared library and returns an opaque handle.
func openLibrary(path string) (uintptr, error) {
	absolute, err := filepath.Abs(path)
	if err != nil {
		return 0, err
	}
	wide, err := syscall.UTF16PtrFromString(absolute)
	if err != nil {
		return 0, err
	}
	handle, _, reason := procLoadLibraryEx.Call(
		uintptr(unsafe.Pointer(wide)), 0, loadWithAlteredSearchPath)
	runtime.KeepAlive(wide)
	if handle == 0 {
		return 0, reason
	}
	return handle, nil
}

// lookupSymbol resolves one exported symbol in an open library.
func lookupSymbol(handle uintptr, name string) (uintptr, error) {
	return syscall.GetProcAddress(syscall.Handle(handle), name)
}

// closeLibrary releases a handle returned by openLibrary.
func closeLibrary(handle uintptr) error {
	return syscall.FreeLibrary(syscall.Handle(handle))
}
