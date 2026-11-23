//go:build !windows && !linux

package encoder

import "github.com/yh2237/gograd/tensor"

var acceleratedDevices []tensor.Device
