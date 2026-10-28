package refmatch

import "fmt"

// FeatureModeは照合に使う特徴量。空なら従来のWavLM検索を使う。
type FeatureMode string

const (
	FeatureWavLM    FeatureMode = "wavlm"
	FeatureCepstrum FeatureMode = "cepstrum"
	CepstrumDims                = 20
)

func (mode FeatureMode) Validate() error {
	if mode != "" && mode != FeatureWavLM && mode != FeatureCepstrum {
		return fmt.Errorf("unsupported feature mode %q (want wavlm or cepstrum)", mode)
	}
	return nil
}

// cepstrumFromCodesは保存用の60次元の符号から検索係数を取り出す。復号は不要。
// 音量を表すC0を除き、C1〜C20を等重みで平均してL2正規化する。
// 検索距離は1-cos(a,b)。前後の文脈、有声・無声の罰則はWavLM検索と共通。
func cepstrumFromCodes(codes []float32, frames int) []float32 {
	if frames <= 0 || len(codes) != frames*subFrames*spectrumDims {
		return nil
	}
	values := make([]float32, frames*CepstrumDims)
	for frame := range frames {
		row := values[frame*CepstrumDims : (frame+1)*CepstrumDims]
		for sub := range subFrames {
			start := (frame*subFrames+sub)*spectrumDims + 1
			for d := range CepstrumDims {
				row[d] += codes[start+d] / subFrames
			}
		}
		copy(row, normalized(row))
	}
	return values
}

// ReferenceCepstrumはWORLDの包絡を索引と同じC1〜C20へ変換する。
// frameMSは渡した包絡の周期。索引と同じ標本化周波数で解析した包絡を渡す。
// 20ms周期なら各行、5ms周期なら4行の平均を1検索フレームにする。
func ReferenceCepstrum(spectrum []float64, sampleRate, fftSize int, frameMS float64) ([]float32, int, error) {
	if sampleRate <= 0 || fftSize < 2 || (frameMS != FrameMS && frameMS != worldFrameMS) {
		return nil, 0, fmt.Errorf("invalid WORLD spectrum parameters")
	}
	bins := fftSize/2 + 1
	if len(spectrum) == 0 || len(spectrum)%bins != 0 {
		return nil, 0, fmt.Errorf("invalid WORLD spectrum dimensions")
	}
	rowsPerFrame := int(FrameMS / frameMS)
	frames := len(spectrum) / bins / rowsPerFrame
	if frames == 0 {
		return nil, 0, fmt.Errorf("WORLD spectrum has no complete frames")
	}
	c := newCodec(sampleRate, fftSize, spectrumDims)
	code := make([]float64, spectrumDims)
	out := make([]float32, frames*CepstrumDims)
	for frame := range frames {
		row := out[frame*CepstrumDims : (frame+1)*CepstrumDims]
		for sub := range rowsPerFrame {
			start := (frame*rowsPerFrame + sub) * bins
			c.codeSpectrum(spectrum[start:start+bins], code)
			for d := range CepstrumDims {
				// 保存した包絡と同じfloat16精度へ揃える。
				row[d] += halfToFloat32(float32ToHalf(float32(code[d+1]))) / float32(rowsPerFrame)
			}
		}
		copy(row, normalized(row))
	}
	return out, frames, nil
}

func (index *Index) cepstrumIndex() (*Index, error) {
	values := index.Cepstrum
	if len(values) == 0 {
		// 第6版も既存の包絡を再利用できる。WavLMの値や検索キャッシュは変更しない。
		values = cepstrumFromCodes(index.Spectrum, index.Frames)
	}
	if len(values) != index.Frames*CepstrumDims || index.Frames <= 0 {
		return nil, fmt.Errorf("index has no usable cepstrum")
	}
	matched := index.WithFeatures(values)
	matched.Dims = CepstrumDims
	return matched, nil
}
