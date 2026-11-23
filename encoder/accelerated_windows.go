//go:build windows

package encoder

import "github.com/yh2237/gograd/tensor"

var acceleratedDevices = []tensor.Device{tensor.CUDA, tensor.DirectML, tensor.Vulkan}
