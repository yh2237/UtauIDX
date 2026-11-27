package index

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"testing"

	"github.com/yh2237/UtauIDX/encoder"
	"github.com/yh2237/UtauIDX/speechpool"
)

func TestPartCacheReusesWorldAndExistingChannels(t *testing.T) {
	dir := t.TempDir()
	path := testWAV(t, dir, "recording.wav", 16000, 160, 1)
	cache := OpenPartCache(filepath.Join(dir, "parts"))
	world := &testWorld{}
	a, b := &testEncoder{id: "ChannelA", dims: 8, frames: 8}, &testEncoder{id: "ChannelB", dims: 4, frames: 7}
	entries := []speechpool.Entry{{Path: path}}
	opts := BuildOptions{Cache: cache}
	first, err := Build(entries, world, []encoder.Encoder{a}, opts)
	if err != nil {
		t.Fatal(err)
	}
	if world.calls.Swap(0) != 1 || a.calls.Swap(0) != 1 {
		t.Fatal("initial Build did not analyze/extract once")
	}
	wavBytes, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct{ prefix, suffix, magic string }{
		{"world\x00v1\x0016000\x000\x00160\x00", ".world", worldMagic},
		{"chan\x00v1\x00ChannelA\x00", ".chan", chanMagic},
	} {
		key := fmt.Sprintf("%x", sha256.Sum256(append([]byte(tc.prefix), wavBytes...)))
		data, err := os.ReadFile(filepath.Join(cache.dir, key[:2], key+tc.suffix))
		if err != nil || !bytes.HasPrefix(data, []byte(tc.magic)) {
			t.Fatalf("%s preamble = %q, error = %v", tc.suffix, data[:min(len(data), 5)], err)
		}
	}
	both, err := Build(entries, world, []encoder.Encoder{a, b}, opts)
	if err != nil {
		t.Fatal(err)
	}
	if world.calls.Load() != 0 || a.calls.Load() != 0 || b.calls.Swap(0) != 1 {
		t.Fatal("adding a model to Build missed an independent cache")
	}
	if !slices.Equal(first.Position, both.Position) || both.Position[both.Frames-1] != 7 {
		t.Fatalf("positions = %v and %v, want retained trailing frame at 7", first.Position, both.Position)
	}
	packed, _ := first.PackedChannel(a.ID())
	if err := AddChannel(first, b, cache, nil); err != nil {
		t.Fatal(err)
	}
	repacked, _ := first.PackedChannel(a.ID())
	if repacked == packed || b.calls.Load() != 0 || !reflect.DeepEqual(first.Header, both.Header) ||
		!reflect.DeepEqual(first.Channels, both.Channels) {
		t.Fatal("AddChannel did not reuse the cached raw features or invalidate packed matrices")
	}
	replace := &testEncoder{id: b.ID(), dims: 5, frames: 6}
	if err := AddChannel(first, replace, cache, nil); err != nil {
		t.Fatal(err)
	}
	if replace.calls.Load() != 1 || len(first.Header.Channels) != 2 || first.ChannelDims(b.ID()) != 5 || first.Validate() != nil {
		t.Fatal("AddChannel did not replace the existing channel in place")
	}

	entries[0].Regions = []speechpool.Region{{Offset: 20, Blank: -100}}
	updated, err := Build(entries, world, []encoder.Encoder{a, replace}, opts)
	if err != nil {
		t.Fatal(err)
	}
	if world.calls.Load() != 1 || a.calls.Load() != 0 || replace.calls.Load() != 1 ||
		!slices.Equal(updated.Position, []int32{1, 2, 4, 5}) {
		t.Fatalf("region edit: calls %d/%d/%d, positions %v", world.calls.Load(), a.calls.Load(), replace.calls.Load(), updated.Position)
	}

	worldCacheKey := worldKey(wavBytes, 16000, 20, 120, false)
	if err := os.WriteFile(cache.path(worldCacheKey, "world"), []byte("bad"), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := Build(entries, world, []encoder.Encoder{a, replace}, opts); err != nil || world.calls.Load() != 2 || a.calls.Load() != 0 || replace.calls.Load() != 1 {
		t.Fatalf("Build after WORLD cache corruption returned %v, calls %d/%d/%d", err, world.calls.Load(), a.calls.Load(), replace.calls.Load())
	}
}

func TestPartCacheRejectsInvalidSizes(t *testing.T) {
	world := worldPartHeader{Magic: [5]byte([]byte(worldMagic)), Usable: 1, Rate: 16000, FFTSize: 128, APBands: 1, Frames: maxFrames}
	channel := channelPartHeader{Magic: [5]byte([]byte(chanMagic)), Frames: maxFrames, Dims: maxDims}
	var data bytes.Buffer
	if err := binary.Write(&data, binary.LittleEndian, world); err != nil {
		t.Fatal(err)
	}
	if _, err := decodeWorldPart(bytes.NewReader(data.Bytes()), int64(data.Len())); err == nil {
		t.Fatal("accepted a cache header declaring missing WORLD rows")
	}
	data.Reset()
	if err := binary.Write(&data, binary.LittleEndian, channel); err != nil {
		t.Fatal(err)
	}
	if _, err := decodeChannelPart(bytes.NewReader(data.Bytes()), int64(data.Len()), maxDims); err == nil {
		t.Fatal("accepted a cache header declaring missing channel rows")
	}
}

func TestBuildBatchesExternalInference(t *testing.T) {
	dir := t.TempDir()
	first := speechpool.Entry{Path: testWAV(t, dir, "first.wav", 16000, 160, 1)}
	second := speechpool.Entry{Path: testWAV(t, dir, "second.wav", 16000, 160, 2)}
	entries := []speechpool.Entry{first, second, first}
	store := encoder.NewExternalStore()
	a := store.Encoder("a", 8, FrameMS, nil)
	b := store.Encoder("b", 4, FrameMS, nil)
	encs := []encoder.Encoder{a, b}
	world, opts := &testWorld{}, BuildOptions{Cache: OpenPartCache(filepath.Join(dir, "parts"))}
	_, err := Build(entries, world, encs, opts)
	var missing *encoder.NeedInference
	if !errors.As(err, &missing) || len(missing.Jobs) != 4 {
		t.Fatalf("Build inference error = %v, want 4 unique jobs", err)
	}
	var order []string
	for _, job := range missing.Jobs {
		if len(job.Samples) != 2560 {
			t.Fatalf("job has %d samples, want 2560", len(job.Samples))
		}
		order = append(order, job.EncoderID)
		dims := a.Dims()
		if job.EncoderID == b.ID() {
			dims = b.Dims()
		}
		if err := store.Provide(job.EncoderID, job.Key, testFeatures(8, dims), 8, dims); err != nil {
			t.Fatal(err)
		}
	}
	if !slices.Equal(order, []string{"a", "b", "a", "b"}) || world.calls.Load() != 0 {
		t.Fatalf("job order = %v, WORLD calls = %d", order, world.calls.Load())
	}
	idx, err := Build(entries, world, encs, opts)
	if err != nil {
		t.Fatal(err)
	}
	if err := idx.Validate(); err != nil || len(idx.Files) != 3 || idx.Files[0].Path != first.Path || idx.Files[1].Path != second.Path {
		t.Fatalf("rebuilt files = %v, validation error = %v", idx.Files, err)
	}
	c := store.Encoder("c", 3, FrameMS, nil)
	err = AddChannel(idx, c, opts.Cache, nil)
	if !errors.As(err, &missing) || len(missing.Jobs) != 2 || idx.HasChannel("c") {
		t.Fatalf("AddChannel error = %v, channels = %v", err, idx.Header.Channels)
	}
	for _, job := range missing.Jobs {
		if err := store.Provide(job.EncoderID, job.Key, testFeatures(6, c.Dims()), 6, c.Dims()); err != nil {
			t.Fatal(err)
		}
	}
	if err := AddChannel(idx, c, opts.Cache, nil); err != nil || !idx.HasChannel("c") {
		t.Fatalf("AddChannel retry = %v, channels = %v", err, idx.Header.Channels)
	}
}
