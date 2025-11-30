/**
 * CrisisTriage AI - Audio Stream Hook
 * 
 * Captures microphone audio and encodes it as PCM 16-bit, 16kHz mono
 * for streaming to the backend via WebSocket.
 * 
 * IMPORTANT SAFETY NOTICE:
 *   This is a RESEARCH AND SIMULATION tool only.
 *   NOT a medical device. NOT suitable for real crisis intervention.
 *   Audio is processed by local/controlled backends only.
 * 
 * Audio Format Output:
 *   - PCM 16-bit signed integer
 *   - 16000 Hz sample rate
 *   - Mono (single channel)
 *   - Little-endian byte order
 * 
 * Usage:
 *   const { isRecording, error, startRecording, stopRecording } = useAudioStream({
 *     onAudioChunk: (chunk) => sendAudioChunk(chunk),
 *   });
 */

'use client';

import { useState, useRef, useCallback, useEffect } from 'react';

// Target audio format for backend
const TARGET_SAMPLE_RATE = 16000;
const CHUNK_INTERVAL_MS = 250; // Send audio chunks every 250ms

export interface UseAudioStreamOptions {
  /** Callback when a new audio chunk is ready */
  onAudioChunk: (chunk: ArrayBuffer) => void;
  
  /** Callback when recording starts */
  onStart?: () => void;
  
  /** Callback when recording stops */
  onStop?: () => void;
  
  /** Callback on error */
  onError?: (error: Error) => void;
  
  /** Interval for sending chunks in milliseconds (default: 250) */
  chunkIntervalMs?: number;
}

export interface UseAudioStreamReturn {
  /** Whether currently recording */
  isRecording: boolean;
  
  /** Whether microphone permission is granted */
  hasPermission: boolean | null;
  
  /** Error message if any */
  error: string | null;
  
  /** Audio level (0-1) for visualization */
  audioLevel: number;
  
  /** Start recording from microphone */
  startRecording: () => Promise<void>;
  
  /** Stop recording */
  stopRecording: () => void;
  
  /** Request microphone permission without starting */
  requestPermission: () => Promise<boolean>;
}

/**
 * Hook for capturing and streaming microphone audio.
 */
export function useAudioStream(options: UseAudioStreamOptions): UseAudioStreamReturn {
  const {
    onAudioChunk,
    onStart,
    onStop,
    onError,
    chunkIntervalMs = CHUNK_INTERVAL_MS,
  } = options;

  // State
  const [isRecording, setIsRecording] = useState(false);
  const [hasPermission, setHasPermission] = useState<boolean | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [audioLevel, setAudioLevel] = useState(0);

  // Refs for audio processing
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const chunkBufferRef = useRef<Float32Array[]>([]);
  const chunkIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const levelIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  // Use ref to track recording state for callbacks (avoids stale closure)
  const isRecordingRef = useRef<boolean>(false);

  /**
   * Request microphone permission.
   */
  const requestPermission = useCallback(async (): Promise<boolean> => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
          sampleRate: { ideal: TARGET_SAMPLE_RATE },
          channelCount: { exact: 1 },
        },
      });

      // Stop the stream immediately - we just wanted to check permission
      stream.getTracks().forEach(track => track.stop());

      setHasPermission(true);
      setError(null);
      return true;
    } catch (err) {
      console.error('Microphone permission denied:', err);
      setHasPermission(false);
      
      if (err instanceof Error) {
        if (err.name === 'NotAllowedError') {
          setError('Microphone access denied. Please grant permission in your browser settings.');
        } else if (err.name === 'NotFoundError') {
          setError('No microphone found. Please connect a microphone.');
        } else {
          setError(`Microphone error: ${err.message}`);
        }
        onError?.(err);
      }
      return false;
    }
  }, [onError]);

  /**
   * Convert Float32Array audio samples to Int16Array (PCM 16-bit).
   */
  const floatTo16BitPCM = useCallback((float32Array: Float32Array): Int16Array => {
    const int16Array = new Int16Array(float32Array.length);
    for (let i = 0; i < float32Array.length; i++) {
      // Clamp to [-1, 1] range
      const s = Math.max(-1, Math.min(1, float32Array[i]));
      // Convert to 16-bit integer
      int16Array[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
    }
    return int16Array;
  }, []);

  /**
   * Resample audio from source sample rate to target sample rate.
   */
  const resample = useCallback((
    samples: Float32Array,
    sourceSampleRate: number,
    targetSampleRate: number
  ): Float32Array => {
    if (sourceSampleRate === targetSampleRate) {
      return samples;
    }

    const ratio = sourceSampleRate / targetSampleRate;
    const newLength = Math.floor(samples.length / ratio);
    const result = new Float32Array(newLength);

    for (let i = 0; i < newLength; i++) {
      const sourceIndex = i * ratio;
      const sourceIndexFloor = Math.floor(sourceIndex);
      const sourceIndexCeil = Math.min(sourceIndexFloor + 1, samples.length - 1);
      const fraction = sourceIndex - sourceIndexFloor;

      // Linear interpolation
      result[i] = samples[sourceIndexFloor] * (1 - fraction) + samples[sourceIndexCeil] * fraction;
    }

    return result;
  }, []);

  /**
   * Process buffered audio chunks and send to callback.
   */
  const processAndSendChunks = useCallback(() => {
    if (chunkBufferRef.current.length === 0) {
      console.log('[Audio] No chunks to process');
      return;
    }

    // Combine all buffered chunks
    const totalLength = chunkBufferRef.current.reduce((acc: number, chunk: Float32Array) => acc + chunk.length, 0);
    const combined = new Float32Array(totalLength);
    let offset = 0;
    for (const chunk of chunkBufferRef.current) {
      combined.set(chunk, offset);
      offset += chunk.length;
    }

    // Clear buffer
    chunkBufferRef.current = [];

    // Get audio context sample rate
    const sourceSampleRate = audioContextRef.current?.sampleRate || 44100;

    // Resample to target rate
    const resampled = resample(combined, sourceSampleRate, TARGET_SAMPLE_RATE);

    // Convert to PCM 16-bit
    const pcm16 = floatTo16BitPCM(resampled);

    console.log('[Audio] Sending chunk:', pcm16.byteLength, 'bytes, samples:', resampled.length, 'source rate:', sourceSampleRate);

    // Send as ArrayBuffer (cast to avoid TypeScript strict type check)
    onAudioChunk(pcm16.buffer as ArrayBuffer);
  }, [onAudioChunk, resample, floatTo16BitPCM]);

  /**
   * Update audio level for visualization.
   */
  const updateAudioLevel = useCallback(() => {
    if (!analyserRef.current) return;

    const dataArray = new Uint8Array(analyserRef.current.fftSize);
    analyserRef.current.getByteTimeDomainData(dataArray);

    // Calculate RMS level
    let sum = 0;
    for (let i = 0; i < dataArray.length; i++) {
      const normalized = (dataArray[i] - 128) / 128;
      sum += normalized * normalized;
    }
    const rms = Math.sqrt(sum / dataArray.length);
    
    // Apply some smoothing and scaling
    setAudioLevel(Math.min(1, rms * 3));
  }, []);

  /**
   * Start recording from microphone.
   */
  const startRecording = useCallback(async () => {
    if (isRecording) return;

    setError(null);

    try {
      // Request microphone access
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
          sampleRate: { ideal: TARGET_SAMPLE_RATE },
          channelCount: { exact: 1 },
        },
      });

      mediaStreamRef.current = stream;
      setHasPermission(true);

      // Log stream info for debugging
      const audioTracks = stream.getAudioTracks();
      console.log('[Audio] Got MediaStream with', audioTracks.length, 'audio tracks');
      audioTracks.forEach((track, i) => {
        console.log(`[Audio] Track ${i}:`, track.label, 'enabled:', track.enabled, 'muted:', track.muted, 'readyState:', track.readyState);
      });

      // Create audio context
      const audioContext = new (window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext)();
      audioContextRef.current = audioContext;

      // Resume audio context if suspended (browser autoplay policy)
      if (audioContext.state === 'suspended') {
        console.log('[Audio] AudioContext suspended, resuming...');
        await audioContext.resume();
        console.log('[Audio] AudioContext resumed, state:', audioContext.state);
      }

      console.log('[Audio] AudioContext created, state:', audioContext.state, 'sampleRate:', audioContext.sampleRate);

      // Create source from stream
      const source = audioContext.createMediaStreamSource(stream);
      sourceRef.current = source;

      // Create analyser for level visualization
      const analyser = audioContext.createAnalyser();
      analyser.fftSize = 256;
      analyserRef.current = analyser;
      source.connect(analyser);

      // Create processor for capturing audio data
      // Using ScriptProcessorNode (deprecated but widely supported)
      // In future, migrate to AudioWorklet for better performance
      const bufferSize = 4096;
      const processor = audioContext.createScriptProcessor(bufferSize, 1, 1);
      processorRef.current = processor;

      let processCallCount = 0;
      processor.onaudioprocess = (event) => {
        processCallCount++;
        
        // Log first few calls to confirm it's working
        if (processCallCount <= 3) {
          console.log('[Audio] onaudioprocess fired #', processCallCount, 'isRecording:', isRecordingRef.current);
        }
        
        // Use ref instead of state to avoid stale closure
        if (!isRecordingRef.current) {
          if (processCallCount <= 5) {
            console.log('[Audio] onaudioprocess skipped - not recording');
          }
          return;
        }

        const inputData = event.inputBuffer.getChannelData(0);
        
        // Check if there's actual audio data (not silence)
        let maxVal = 0;
        for (let i = 0; i < inputData.length; i++) {
          const absVal = Math.abs(inputData[i]);
          if (absVal > maxVal) maxVal = absVal;
        }
        
        // Copy the data (input buffer will be reused)
        const chunk = new Float32Array(inputData.length);
        chunk.set(inputData);
        chunkBufferRef.current.push(chunk);
        
        // Debug: log every 10th chunk with max value
        if (chunkBufferRef.current.length % 10 === 0) {
          console.log('[Audio] Captured chunk #', chunkBufferRef.current.length, 'max amplitude:', maxVal.toFixed(4));
        }
      };

      // Connect nodes
      source.connect(processor);
      processor.connect(audioContext.destination);

      console.log('[Audio] Audio graph connected: source -> processor -> destination');

      // Monitor for stream ending
      stream.getAudioTracks().forEach(track => {
        track.onended = () => {
          console.log('[Audio] Audio track ended unexpectedly!');
        };
        track.onmute = () => {
          console.log('[Audio] Audio track muted!');
        };
      });

      // Start chunk sending interval
      chunkIntervalRef.current = setInterval(processAndSendChunks, chunkIntervalMs);

      // Start level update interval
      levelIntervalRef.current = setInterval(updateAudioLevel, 50);

      // Set recording state (both ref and state)
      isRecordingRef.current = true;
      setIsRecording(true);
      onStart?.();

      console.log('Audio recording started', {
        sampleRate: audioContext.sampleRate,
        targetRate: TARGET_SAMPLE_RATE,
      });

    } catch (err) {
      console.error('Failed to start recording:', err);
      
      if (err instanceof Error) {
        if (err.name === 'NotAllowedError') {
          setError('Microphone access denied. Please grant permission.');
          setHasPermission(false);
        } else if (err.name === 'NotFoundError') {
          setError('No microphone found.');
        } else {
          setError(`Recording error: ${err.message}`);
        }
        onError?.(err);
      }
    }
  }, [isRecording, chunkIntervalMs, processAndSendChunks, updateAudioLevel, onStart, onError]);

  /**
   * Stop recording.
   */
  const stopRecording = useCallback(() => {
    if (!isRecordingRef.current) return;

    // Log why we're stopping (stack trace for debugging)
    console.log('[Audio] stopRecording called');
    console.trace('[Audio] Stop recording stack trace');

    // Stop recording immediately via ref
    isRecordingRef.current = false;

    // Clear intervals
    if (chunkIntervalRef.current) {
      clearInterval(chunkIntervalRef.current);
      chunkIntervalRef.current = null;
    }
    if (levelIntervalRef.current) {
      clearInterval(levelIntervalRef.current);
      levelIntervalRef.current = null;
    }

    // Send any remaining buffered audio
    processAndSendChunks();

    // Disconnect and clean up audio nodes
    if (processorRef.current) {
      processorRef.current.disconnect();
      processorRef.current.onaudioprocess = null;
      processorRef.current = null;
    }
    if (sourceRef.current) {
      sourceRef.current.disconnect();
      sourceRef.current = null;
    }
    if (analyserRef.current) {
      analyserRef.current.disconnect();
      analyserRef.current = null;
    }
    if (audioContextRef.current) {
      audioContextRef.current.close();
      audioContextRef.current = null;
    }

    // Stop media stream tracks
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((track: MediaStreamTrack) => track.stop());
      mediaStreamRef.current = null;
    }

    // Clear buffer
    chunkBufferRef.current = [];

    setIsRecording(false);
    setAudioLevel(0);
    onStop?.();

    console.log('Audio recording stopped');
  }, [processAndSendChunks, onStop]);

  // Cleanup on unmount - use empty deps to only run on actual unmount
  // We use refs directly to avoid stale closure issues
  useEffect(() => {
    return () => {
      console.log('[Audio] Component unmounting, cleaning up...');
      if (isRecordingRef.current) {
        // Inline cleanup instead of calling stopRecording to avoid dependency issues
        isRecordingRef.current = false;
        
        if (chunkIntervalRef.current) {
          clearInterval(chunkIntervalRef.current);
          chunkIntervalRef.current = null;
        }
        if (levelIntervalRef.current) {
          clearInterval(levelIntervalRef.current);
          levelIntervalRef.current = null;
        }
        if (processorRef.current) {
          processorRef.current.disconnect();
          processorRef.current = null;
        }
        if (sourceRef.current) {
          sourceRef.current.disconnect();
          sourceRef.current = null;
        }
        if (analyserRef.current) {
          analyserRef.current.disconnect();
          analyserRef.current = null;
        }
        if (audioContextRef.current) {
          audioContextRef.current.close();
          audioContextRef.current = null;
        }
        if (mediaStreamRef.current) {
          mediaStreamRef.current.getTracks().forEach((track: MediaStreamTrack) => track.stop());
          mediaStreamRef.current = null;
        }
        chunkBufferRef.current = [];
      }
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []); // Empty deps - only run on actual unmount

  return {
    isRecording,
    hasPermission,
    error,
    audioLevel,
    startRecording,
    stopRecording,
    requestPermission,
  };
}

export default useAudioStream;
