//go:build linux || darwin || freebsd

package motionbricks

import "github.com/ebitengine/purego"

// openLibrary loads a shared library and returns an opaque handle.
func openLibrary(path string) (uintptr, error) {
	return purego.Dlopen(path, purego.RTLD_NOW|purego.RTLD_LOCAL)
}

// lookupSymbol resolves one exported symbol in an open library.
func lookupSymbol(handle uintptr, name string) (uintptr, error) {
	return purego.Dlsym(handle, name)
}

// closeLibrary releases a handle returned by openLibrary.
func closeLibrary(handle uintptr) error {
	return purego.Dlclose(handle)
}
