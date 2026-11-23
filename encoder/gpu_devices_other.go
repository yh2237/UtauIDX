//go:build !windows && !linux

package encoder

import (
	"github.com/yh2237/gograd/gpu"
	"github.com/yh2237/gograd/tensor"
)

func enterGPUDevice(tensor.Device) (func(), error) {
	return nil, gpu.ErrUnavailable
}
