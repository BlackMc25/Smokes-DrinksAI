
import { GoogleGenAI, LiveServerMessage, Modality } from "@google/genai";

export class LiveVoiceService {
  private ai: GoogleGenAI;
  private session: any = null;
  private audioContext: AudioContext | null = null;
  private processor: ScriptProcessorNode | null = null;
  private source: MediaStreamAudioSourceNode | null = null;
  private stream: MediaStream | null = null;
  private nextStartTime: number = 0;

  constructor() {
    this.ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY || "" });
  }

  async start(callbacks: {
    onMessage?: (text: string) => void;
    onInterrupted?: () => void;
    onError?: (error: any) => void;
    onClose?: () => void;
  }) {
    try {
      this.audioContext = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 16000 });
      this.nextStartTime = this.audioContext.currentTime;

      this.session = await this.ai.live.connect({
        model: "gemini-2.5-flash-native-audio-preview-09-2025",
        callbacks: {
          onopen: () => {
            this.startMic();
          },
          onmessage: async (message: LiveServerMessage) => {
            if (message.serverContent?.modelTurn?.parts) {
              for (const part of message.serverContent.modelTurn.parts) {
                if (part.inlineData) {
                  this.playAudio(part.inlineData.data);
                }
                if (part.text && callbacks.onMessage) {
                  callbacks.onMessage(part.text);
                }
              }
            }
            
            if (message.serverContent?.interrupted) {
              this.stopPlayback();
              if (callbacks.onInterrupted) callbacks.onInterrupted();
            }

            if (message.serverContent?.turnComplete) {
              // Turn complete
            }
          },
          onerror: (error) => {
            console.error("Live API Error:", error);
            if (callbacks.onError) callbacks.onError(error);
          },
          onclose: () => {
            this.stopMic();
            if (callbacks.onClose) callbacks.onClose();
          },
        },
        config: {
          responseModalities: [Modality.AUDIO],
          speechConfig: {
            voiceConfig: { prebuiltVoiceConfig: { voiceName: "Zephyr" } },
          },
          systemInstruction: "Your name is Elena. You are a helpful health assistant. You are in a real-time voice conversation. Be concise, friendly, and supportive. Focus on health monitoring and behavioral advice for smoking and drinking.",
        },
      });
    } catch (error) {
      console.error("Failed to connect to Live API:", error);
      throw error;
    }
  }

  private async startMic() {
    try {
      this.stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      this.source = this.audioContext!.createMediaStreamSource(this.stream);
      this.processor = this.audioContext!.createScriptProcessor(4096, 1, 1);

      this.processor.onaudioprocess = (e) => {
        const inputData = e.inputBuffer.getChannelData(0);
        // Convert Float32 to Int16 PCM
        const pcmData = new Int16Array(inputData.length);
        for (let i = 0; i < inputData.length; i++) {
          pcmData[i] = Math.max(-1, Math.min(1, inputData[i])) * 0x7FFF;
        }
        
        // Convert to Base64
        const base64Data = btoa(String.fromCharCode(...new Uint8Array(pcmData.buffer)));
        
        if (this.session) {
          this.session.sendRealtimeInput({
            media: { data: base64Data, mimeType: 'audio/pcm;rate=16000' }
          });
        }
      };

      this.source.connect(this.processor);
      this.processor.connect(this.audioContext!.destination);
    } catch (error) {
      console.error("Error starting microphone:", error);
    }
  }

  private playAudio(base64Data: string) {
    if (!this.audioContext) return;

    const binaryString = atob(base64Data);
    const bytes = new Uint8Array(binaryString.length);
    for (let i = 0; i < binaryString.length; i++) {
      bytes[i] = binaryString.charCodeAt(i);
    }
    
    const pcmData = new Int16Array(bytes.buffer);
    const floatData = new Float32Array(pcmData.length);
    for (let i = 0; i < pcmData.length; i++) {
      floatData[i] = pcmData[i] / 0x7FFF;
    }

    const buffer = this.audioContext.createBuffer(1, floatData.length, 24000); // Model output is 24kHz
    buffer.getChannelData(0).set(floatData);

    const source = this.audioContext.createBufferSource();
    source.buffer = buffer;
    source.connect(this.audioContext.destination);
    
    const startTime = Math.max(this.audioContext.currentTime, this.nextStartTime);
    source.start(startTime);
    this.nextStartTime = startTime + buffer.duration;
  }

  private stopPlayback() {
    // In a more robust implementation, we'd keep track of sources and stop them
    this.nextStartTime = this.audioContext?.currentTime || 0;
  }

  private stopMic() {
    if (this.processor) {
      this.processor.disconnect();
      this.processor = null;
    }
    if (this.source) {
      this.source.disconnect();
      this.source = null;
    }
    if (this.stream) {
      this.stream.getTracks().forEach(track => track.stop());
      this.stream = null;
    }
  }

  stop() {
    if (this.session) {
      this.session.close();
      this.session = null;
    }
    this.stopMic();
    if (this.audioContext) {
      this.audioContext.close();
      this.audioContext = null;
    }
  }
}
