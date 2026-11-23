package audio

import (
	"errors"
	"io"
	"os"

	"github.com/yh2237/audiodsp/wav"

	"github.com/yh2237/UtauIDX/atomicfile"
)

type PCM = wav.PCM

func ReadWav(path string) (*PCM, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	return DecodeWav(file)
}

func DecodeWav(input io.Reader) (*PCM, error) {
	return wav.Decode(input)
}

func WriteWav(path string, pcm *PCM) error {
	if err := validatePCM(pcm); err != nil {
		return err
	}
	if len(pcm.Data) == 0 {
		return errors.New("empty pcm data")
	}

	return atomicfile.Write(path, func(file io.Writer) error {
		return wav.Encode(file, pcm)
	})
}

// 合成結果用。サンプルレート・チャンネル数が不正な場合はnilを返す。
func PCMToWavBytes(pcm *PCM) []byte {
	if err := validatePCM(pcm); err != nil {
		return nil
	}
	data, err := wav.Bytes(pcm)
	if err != nil {
		return nil
	}
	return data
}

// validatePCMは、WAVにできるPCMの形を確かめる。チャンネル数で割り切れない
// サンプル数は、書き出すと読み直せないWAVになるため受け付けない。
func validatePCM(pcm *PCM) error {
	if pcm == nil {
		return errors.New("nil pcm")
	}
	if pcm.SampleRate <= 0 || pcm.Channels <= 0 {
		return errors.New("invalid pcm metadata")
	}
	if len(pcm.Data)%pcm.Channels != 0 {
		return errors.New("pcm data is not aligned to channels")
	}
	return nil
}
