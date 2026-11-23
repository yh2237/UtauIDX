//go:build windows

package encoder

import (
	"runtime"

	"github.com/yh2237/gograd/cuda"
	_ "github.com/yh2237/gograd/directml"
	"github.com/yh2237/gograd/gpu"
	"github.com/yh2237/gograd/tensor"
	_ "github.com/yh2237/gograd/vulkan"
)

func enterGPUDevice(device tensor.Device) (func(), error) {
	switch device {
	case tensor.CUDA:
		if !cuda.Available() {
			return nil, cuda.ErrUnavailable
		}
		runtime.LockOSThread()
		if err := cuda.SetDevice(0); err != nil {
			runtime.UnlockOSThread()
			return nil, err
		}
		return runtime.UnlockOSThread, nil
	case tensor.DirectML, tensor.Vulkan:
		if gpu.Available(string(device)) {
			return func() {}, nil
		}
	}
	return nil, gpu.ErrUnavailable
}
