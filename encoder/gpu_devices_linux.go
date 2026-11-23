//go:build linux

package encoder

import (
	"github.com/yh2237/gograd/gpu"
	"github.com/yh2237/gograd/tensor"
	_ "github.com/yh2237/gograd/vulkan"
)

func enterGPUDevice(device tensor.Device) (func(), error) {
	if device == tensor.Vulkan && gpu.Available("vulkan") {
		return func() {}, nil
	}
	return nil, gpu.ErrUnavailable
}
