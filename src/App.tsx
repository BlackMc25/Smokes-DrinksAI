
import React, { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { 
  MessageSquare,
  Mic, 
  MicOff, 
  Activity, 
  ShieldAlert, 
  Heart, 
  Send, 
  Volume2,
  ChevronRight,
  ChevronLeft,
  Info,
  User as UserIcon,
  Droplets,
  Wind,
  CheckCircle2,
  AlertTriangle,
  X,
  BarChart3,
  TrendingUp,
  Star,
  LogIn,
  UserPlus,
  Stethoscope,
  Eye,
  EyeOff
} from 'lucide-react';
import Markdown from 'react-markdown';
import { 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell
} from 'recharts';
import { loadModel, XGBPredictor } from './services/xgbService';
import { getHealthInsights, HealthMetrics } from './services/healthService';
import { GoogleGenAI } from "@google/genai";
import { LiveVoiceService } from './services/liveVoiceService';
import { 
  auth, 
  googleProvider, 
  appleProvider,
  signInWithPopup, 
  signOut, 
  onAuthStateChanged, 
  User,
  createUserWithEmailAndPassword,
  signInWithEmailAndPassword,
  updateProfile,
  sendEmailVerification
} from './firebase';

interface Message {
  role: 'user' | 'model';
  text: string;
  timestamp: Date;
  status?: 'sent' | 'delivered' | 'read';
}

class ErrorBoundary extends React.Component<{ children: React.ReactNode }, { hasError: boolean, errorInfo: string | null }> {
  constructor(props: { children: React.ReactNode }) {
    super(props);
    this.state = { hasError: false, errorInfo: null };
  }

  static getDerivedStateFromError(error: any) {
    return { hasError: true, errorInfo: error.message };
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="min-h-screen flex items-center justify-center p-6 bg-black text-white relative">
          {/* App Logo & Name */}
          <div className="absolute top-0 left-0 px-6 py-4 flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-orange-500 flex items-center justify-center shadow-lg shadow-orange-500/20">
              <Activity className="text-white" size={24} />
            </div>
            <span className="text-lg font-bold tracking-tight text-white">
              Smoke&DrinkRisk.AI
            </span>
          </div>

          <div className="glass-card p-8 max-w-md w-full space-y-6 text-center">
            <div className="w-16 h-16 rounded-full bg-red-500/10 flex items-center justify-center text-red-500 mx-auto">
              <AlertTriangle size={32} />
            </div>
            <h2 className="text-2xl font-bold">Something went wrong</h2>
            <p className="text-white/60 text-sm">
              {this.state.errorInfo?.startsWith('{') ? 'A database error occurred. Please try again later.' : 'An unexpected error occurred.'}
            </p>
            <button 
              onClick={() => window.location.reload()}
              className="w-full py-3 bg-orange-500 rounded-xl font-bold hover:bg-orange-600 transition-all"
            >
              Reload Application
            </button>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}

interface AuthModalProps {
  isOpen: boolean;
  onClose: () => void;
  mode: 'signin' | 'signup';
  onModeChange: (mode: 'signin' | 'signup') => void;
  onGoogleSignIn: () => void;
  onAppleSignIn: () => void;
}

function AuthModal({ isOpen, onClose, mode, onModeChange, onGoogleSignIn, onAppleSignIn }: AuthModalProps) {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [displayName, setDisplayName] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [verificationSent, setVerificationSent] = useState(false);
  const [showPassword, setShowPassword] = useState(false);
  const [resendLoading, setResendLoading] = useState(false);
  const [resendStatus, setResendStatus] = useState<'idle' | 'success' | 'error'>('idle');

  const handleResendEmail = async () => {
    const currentUser = auth.currentUser;
    if (!currentUser) {
      setError("Please sign in to resend the verification link.");
      return;
    }
    
    setResendLoading(true);
    setResendStatus('idle');
    try {
      await sendEmailVerification(currentUser);
      setResendStatus('success');
      setTimeout(() => setResendStatus('idle'), 5000);
    } catch (err: any) {
      console.error("Resend error:", err);
      setResendStatus('error');
      setError("Failed to send verification link. Please try again later.");
    } finally {
      setResendLoading(false);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setLoading(true);

    try {
      if (mode === 'signup') {
        const userCredential = await createUserWithEmailAndPassword(auth, email, password);
        await updateProfile(userCredential.user, { displayName });
        
        // Send verification link via Firebase
        try {
          await sendEmailVerification(userCredential.user);
          setVerificationSent(true);
        } catch (emailErr: any) {
          console.error("Initial verification link error:", emailErr);
          setError("Account created, but failed to send verification link. Please sign in to resend it.");
        }
        
        await signOut(auth);
      } else {
        const userCredential = await signInWithEmailAndPassword(auth, email, password);
        const user = userCredential.user;
        
        if (!user.emailVerified) {
          // Send new verification link
          try {
            await sendEmailVerification(user);
          } catch (e) {
            console.error("Signin verification link error:", e);
          }
          
          await signOut(auth);
          setVerificationSent(true);
          setError("Account not verified. We've sent a new verification link to your email.");
        } else {
          onClose();
        }
      }
    } catch (err: any) {
      console.error("Auth error:", err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  if (verificationSent) {
    return (
      <AnimatePresence>
        {isOpen && (
          <div className="fixed inset-0 z-[100] flex items-center justify-center p-4 sm:p-6">
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={onClose}
              className="absolute inset-0 bg-black/60 backdrop-blur-sm"
            />
            <motion.div
              initial={{ opacity: 0, scale: 0.9, y: 20 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.9, y: 20 }}
              className="relative w-full max-w-md glass-card p-6 sm:p-8 space-y-6 text-center"
            >
              <div className="w-16 h-16 rounded-full bg-orange-500/10 flex items-center justify-center text-orange-500 mx-auto">
                <ShieldAlert size={32} />
              </div>
              <div className="space-y-2">
                <h2 className="text-2xl font-bold text-white">Verify Your Email</h2>
                <p className="text-white/60">
                  We've sent a verification link to <span className="text-white font-medium">{email}</span>.
                </p>
                <p className="text-sm text-white/40">
                  Please click the link in the email to verify your account, then return here to sign in.
                </p>
              </div>

              <div className="pt-4 space-y-4">
                <button
                  onClick={() => {
                    setVerificationSent(false);
                    onModeChange('signin');
                  }}
                  className="w-full py-4 bg-orange-500 text-white rounded-xl font-bold hover:bg-orange-600 transition-all shadow-lg shadow-orange-500/20"
                >
                  Back to Sign In
                </button>

                <div className="pt-2">
                  <p className="text-xs text-white/40 mb-2">Didn't receive the email?</p>
                  <button
                    onClick={handleResendEmail}
                    disabled={resendLoading}
                    className="text-sm text-orange-500 font-bold hover:underline disabled:opacity-50"
                  >
                    {resendLoading ? 'Sending...' : 'Resend Verification Link'}
                  </button>
                  
                  {resendStatus === 'success' && (
                    <p className="text-[10px] text-green-400 mt-1">New link sent!</p>
                  )}
                  {resendStatus === 'error' && (
                    <p className="text-[10px] text-red-400 mt-1">Failed to resend. Try again later.</p>
                  )}
                </div>
              </div>
            </motion.div>
          </div>
        )}
      </AnimatePresence>
    );
  }

  return (
    <AnimatePresence>
      {isOpen && (
        <div className="fixed inset-0 z-[100] flex items-center justify-center p-4 sm:p-6">
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
            className="absolute inset-0 bg-black/60 backdrop-blur-sm"
          />
          <motion.div
            initial={{ opacity: 0, scale: 0.9, y: 20 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.9, y: 20 }}
            className="relative w-full max-w-md glass-card p-6 sm:p-8 space-y-6 sm:space-y-8 overflow-hidden"
          >
            <button 
              onClick={onClose}
              className="absolute top-4 right-4 text-white/40 hover:text-white transition-colors"
            >
              <X size={24} />
            </button>

            <div className="flex items-center justify-center gap-4 mb-4">
              <div className="w-12 h-12 rounded-xl bg-orange-500 flex items-center justify-center shadow-lg shadow-orange-500/20">
                <Activity className="text-white" size={24} />
              </div>
              <h2 className="text-3xl font-bold">
                {mode === 'signin' ? 'Welcome Back' : 'Create Account'}
              </h2>
            </div>
            <div className="text-center mb-8">
              <p className="text-white/60">
                {mode === 'signin' ? 'Sign in to access your health dashboard' : 'Join us to start your health journey'}
              </p>
            </div>

            <form onSubmit={handleSubmit} className="space-y-4">
              {mode === 'signup' && (
                <div className="space-y-1">
                  <label className="text-xs font-semibold text-white/40 uppercase tracking-wider ml-1">Full Name</label>
                  <input
                    type="text"
                    required
                    value={displayName}
                    onChange={(e) => setDisplayName(e.target.value)}
                    placeholder="John Doe"
                    className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-3 text-white placeholder:text-white/20 focus:outline-none focus:border-orange-500/50 transition-all"
                  />
                </div>
              )}
              <div className="space-y-1">
                <label className="text-xs font-semibold text-white/40 uppercase tracking-wider ml-1">Email Address</label>
                <input
                  type="email"
                  required
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder="name@example.com"
                  className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-3 text-white placeholder:text-white/20 focus:outline-none focus:border-orange-500/50 transition-all"
                />
              </div>
              <div className="space-y-1">
                <label className="text-xs font-semibold text-white/40 uppercase tracking-wider ml-1">Password</label>
                <div className="relative">
                  <input
                    type={showPassword ? "text" : "password"}
                    required
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    placeholder="••••••••"
                    className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-3 text-white placeholder:text-white/20 focus:outline-none focus:border-orange-500/50 transition-all pr-12"
                  />
                  <button
                    type="button"
                    onClick={() => setShowPassword(!showPassword)}
                    className="absolute right-4 top-1/2 -translate-y-1/2 text-white/40 hover:text-white transition-colors"
                  >
                    {showPassword ? <EyeOff size={18} /> : <Eye size={18} />}
                  </button>
                </div>
              </div>

              {error && (
                <div className="p-3 rounded-lg bg-red-500/10 border border-red-500/20 text-red-400 text-xs flex items-center gap-2">
                  <AlertTriangle size={14} />
                  {error}
                </div>
              )}

              <button
                type="submit"
                disabled={loading}
                className="w-full py-4 bg-orange-500 text-white rounded-xl font-bold hover:bg-orange-600 transition-all shadow-lg shadow-orange-500/20 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {loading ? 'Processing...' : (mode === 'signin' ? 'Sign In' : 'Create Account')}
              </button>
            </form>

            <div className="relative">
              <div className="absolute inset-0 flex items-center">
                <div className="w-full border-t border-white/10"></div>
              </div>
              <div className="relative flex justify-center text-xs uppercase">
                <span className="bg-black px-2 text-white/40">Or continue with</span>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-3">
              <button
                onClick={onGoogleSignIn}
                className="w-full py-4 bg-white text-black rounded-xl font-bold flex items-center justify-center gap-3 hover:bg-white/90 transition-all"
              >
                <img src="https://www.google.com/favicon.ico" alt="Google" className="w-5 h-5" />
                Google
              </button>
              <button
                onClick={onAppleSignIn}
                className="w-full py-4 bg-white text-black rounded-xl font-bold flex items-center justify-center gap-3 hover:bg-white/90 transition-all"
              >
                <img src="https://www.apple.com/favicon.ico" alt="Apple" className="w-5 h-5" />
                Apple
              </button>
            </div>

            <p className="text-center text-sm text-white/60">
              {mode === 'signin' ? "Don't have an account? " : "Already have an account? "}
              <button
                onClick={() => onModeChange(mode === 'signin' ? 'signup' : 'signin')}
                className="text-orange-500 font-bold hover:underline"
              >
                {mode === 'signin' ? 'Sign Up' : 'Sign In'}
              </button>
            </p>
          </motion.div>
        </div>
      )}
    </AnimatePresence>
  );
}

function ClinicalSlider() {
  const images = [
    { 
      src: "/smoking_images.jpg", 
      alt: "Smoking impact visualization" 
    },
    { 
      src: "/smk_img.jpg", 
      alt: "Clinical research on smoking" 
    },
    { 
      src: "/bagoes-ilhamy-Pi_H3N_qbVQ-unsplash.jpg", 
      alt: "Healthy lifestyle choices" 
    },
    { 
      src: "/julia-taubitz-vN5s_G5SPkY-unsplash.jpg", 
      alt: "Medical consultation" 
    },
    { 
      src: "/vitaly-gariev-NDOYHZ_98Rw-unsplash.jpg", 
      alt: "Advanced health monitoring" 
    }
  ];
  const [currentIndex, setCurrentIndex] = useState(0);

  const nextSlide = () => {
    setCurrentIndex((prev) => (prev + 1) % images.length);
  };

  const prevSlide = () => {
    setCurrentIndex((prev) => (prev - 1 + images.length) % images.length);
  };

  useEffect(() => {
    const timer = setInterval(nextSlide, 4000);
    return () => clearInterval(timer);
  }, [currentIndex]);

  return (
    <div className="relative group">
      <div className="overflow-hidden rounded-2xl md:rounded-3xl aspect-[4/3] md:aspect-video glass-card">
        <AnimatePresence mode="wait">
          <motion.img
            key={currentIndex}
            src={images[currentIndex].src}
            alt={images[currentIndex].alt}
            initial={{ opacity: 0, scale: 1.1 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.9 }}
            transition={{ duration: 0.5 }}
            className="w-full h-full object-cover"
            referrerPolicy="no-referrer"
          />
        </AnimatePresence>
      </div>

      {/* Navigation Buttons */}
      <div className="absolute inset-0 flex items-center justify-between p-4 opacity-0 group-hover:opacity-100 transition-opacity">
        <button 
          onClick={prevSlide}
          className="w-12 h-12 rounded-full bg-black/50 backdrop-blur-md text-white flex items-center justify-center hover:bg-orange-500 transition-colors"
        >
          <ChevronLeft size={24} />
        </button>
        <button 
          onClick={nextSlide}
          className="w-12 h-12 rounded-full bg-black/50 backdrop-blur-md text-white flex items-center justify-center hover:bg-orange-500 transition-colors"
        >
          <ChevronRight size={24} />
        </button>
      </div>

      {/* Indicators */}
      <div className="absolute bottom-6 left-1/2 -translate-x-1/2 flex gap-2">
        {images.map((_, i) => (
          <button
            key={i}
            onClick={() => setCurrentIndex(i)}
            className={`w-2 h-2 rounded-full transition-all ${
              i === currentIndex ? 'w-8 bg-orange-500' : 'bg-white/30 hover:bg-white/50'
            }`}
          />
        ))}
      </div>
    </div>
  );
}

export default function App() {
  const [activeTab, setActiveTab] = useState<'home' | 'chat' | 'predict'>('home');
  const [user, setUser] = useState<User | null>(null);
  const [isAuthReady, setIsAuthReady] = useState(false);
  const [isAuthMenuOpen, setIsAuthMenuOpen] = useState(false);
  const [isAuthModalOpen, setIsAuthModalOpen] = useState(false);
  const [authMode, setAuthMode] = useState<'signin' | 'signup'>('signin');
  const [messages, setMessages] = useState<Message[]>([
    { 
      role: 'model', 
      text: "Hello! I'm Elena, your AI Health Companion. I can help you understand your health risks related to smoking and drinking based on your medical metrics. How can I assist you today?",
      timestamp: new Date(),
      status: 'read'
    }
  ]);
  const [input, setInput] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [isListening, setIsListening] = useState(false);
  
  const [metrics, setMetrics] = useState<HealthMetrics>({
    hemoglobin: 15,
    sex_numeric: 1,
    SBP: 120,
    DBP: 80,
    BLDS: 90,
    gamma_GTP: 25,
    BMI: 24.2,
    waistline: 85,
    HDL_chole: 50,
    LDL_chole: 100,
    triglyceride: 120,
    SMK_stat_type_cd: 1,
    DRK_YN: 0,
    urine_protein: 1
  });

  const [predictionResult, setPredictionResult] = useState<{
    smokingProb: number;
    drinkingProb: number;
    insights: string;
  } | null>(null);
  const [isPredicting, setIsPredicting] = useState(false);

  const [smokingModel, setSmokingModel] = useState<XGBPredictor | null>(null);
  const [drinkingModel, setDrinkingModel] = useState<XGBPredictor | null>(null);

  const chatEndRef = useRef<HTMLDivElement>(null);
  const recognitionRef = useRef<any>(null);
  const liveVoiceRef = useRef<LiveVoiceService | null>(null);
  const [isLiveMode, setIsLiveMode] = useState(false);

  const performanceData = [
    { name: 'Accuracy', value: 89 },
    { name: 'Precision', value: 87 },
    { name: 'Recall', value: 85 },
    { name: 'F1 Score', value: 86 }
  ];

  const modelDistribution = [
    { name: 'Smoking Model', value: 45 },
    { name: 'Drinking Model', value: 55 }
  ];

  const COLORS = ['#f97316', '#3b82f6', '#10b981', '#8b5cf6'];

  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, async (currentUser) => {
      if (currentUser) {
        if (!currentUser.emailVerified) {
          setUser(null);
        } else {
          setUser(currentUser);
        }
      } else {
        setUser(null);
      }
      
      setIsAuthReady(true);
    });
    return () => unsubscribe();
  }, []);

  const handleSignIn = async () => {
    setAuthMode('signin');
    setIsAuthModalOpen(true);
  };

  const handleSignUp = async () => {
    setAuthMode('signup');
    setIsAuthModalOpen(true);
  };

  const handleGoogleSignIn = async () => {
    try {
      await signInWithPopup(auth, googleProvider);
      setIsAuthModalOpen(false);
    } catch (error) {
      console.error("Sign in error:", error);
    }
  };

  const handleAppleSignIn = async () => {
    try {
      await signInWithPopup(auth, appleProvider);
      setIsAuthModalOpen(false);
    } catch (error) {
      console.error("Sign in error:", error);
    }
  };

  const handleSignOut = async () => {
    try {
      await signOut(auth);
    } catch (error) {
      console.error("Sign out error:", error);
    }
  };

  useEffect(() => {
    const initModels = async () => {
      try {
        const sm = await loadModel('/Retrain_smoking_model_xgb.json');
        const dm = await loadModel('/Retrain_Drinking_model_xgb.json');
        setSmokingModel(sm);
        setDrinkingModel(dm);
      } catch (err) {
        console.error("Failed to load models:", err);
      }
    };
    initModels();
  }, []);

  useEffect(() => {
    const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
    if (SpeechRecognition) {
      recognitionRef.current = new SpeechRecognition();
      recognitionRef.current.continuous = false;
      recognitionRef.current.interimResults = false;
      recognitionRef.current.lang = 'en-US';

      recognitionRef.current.onresult = (event: any) => {
        const transcript = event.results[0][0].transcript;
        setInput(transcript);
        setIsListening(false);
      };

      recognitionRef.current.onend = () => setIsListening(false);
      recognitionRef.current.onerror = () => setIsListening(false);
    }
  }, []);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  useEffect(() => {
    return () => {
      if (liveVoiceRef.current) {
        liveVoiceRef.current.stop();
      }
    };
  }, []);

  const handleSendMessage = async (e?: React.FormEvent) => {
    e?.preventDefault();
    if (!input.trim()) return;

    const userMsg = input;
    const timestamp = new Date();
    setInput('');
    setMessages(prev => [...prev, { role: 'user', text: userMsg, timestamp, status: 'sent' }]);
    setIsTyping(true);

    // Simulate delivery status
    setTimeout(() => {
      setMessages(prev => prev.map((m, i) => 
        i === prev.length - 1 && m.role === 'user' ? { ...m, status: 'delivered' } : m
      ));
    }, 800);

    try {
      const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY! });
      const chat = ai.chats.create({
        model: "gemini-3-flash-preview",
        config: {
          systemInstruction: "Your name is Elena. You are a professional health assistant. You help users understand health risks, specifically related to smoking and drinking. You use data-driven insights. Always encourage users to consult real doctors for medical diagnosis. Be empathetic, professional, and concise."
        }
      });
      
      // Mark user message as read when AI starts processing
      setTimeout(() => {
        setMessages(prev => prev.map(m => 
          m.role === 'user' && m.status !== 'read' ? { ...m, status: 'read' } : m
        ));
      }, 1500);

      const response = await chat.sendMessage({ message: userMsg });

      setMessages(prev => [...prev, { 
        role: 'model', 
        text: response.text || 'I apologize, I am having trouble processing that.',
        timestamp: new Date(),
        status: 'read'
      }]);
    } catch (error) {
      console.error(error);
      setMessages(prev => [...prev, { 
        role: 'model', 
        text: 'Sorry, I encountered an error connecting to my brain.',
        timestamp: new Date(),
        status: 'read'
      }]);
    } finally {
      setIsTyping(false);
    }
  };

  const toggleVoice = async () => {
    if (activeTab === 'chat') {
      if (!isLiveMode) {
        try {
          if (!liveVoiceRef.current) {
            liveVoiceRef.current = new LiveVoiceService();
          }
          
          setIsLiveMode(true);
          await liveVoiceRef.current.start({
            onMessage: (text) => {
              setMessages(prev => {
                const lastMsg = prev[prev.length - 1];
                if (lastMsg && lastMsg.role === 'model') {
                  // Append to last model message if it's part of the same turn
                  // For simplicity in this UI, we'll just add a new message if it's a new turn
                  // But Live API sends chunks. Let's handle it simply for now.
                  return [...prev.slice(0, -1), { ...lastMsg, text: lastMsg.text + text }];
                }
                return [...prev, { role: 'model', text, timestamp: new Date(), status: 'read' }];
              });
            },
            onInterrupted: () => {
              setIsTyping(false);
            },
            onError: (err) => {
              console.error("Live Voice Error:", err);
              setIsLiveMode(false);
            },
            onClose: () => {
              setIsLiveMode(false);
            }
          });
        } catch (err) {
          console.error("Failed to start live voice:", err);
          setIsLiveMode(false);
        }
      } else {
        if (liveVoiceRef.current) {
          liveVoiceRef.current.stop();
        }
        setIsLiveMode(false);
      }
      return;
    }

    if (!recognitionRef.current) return;
    if (!isListening) {
      recognitionRef.current.start();
      setIsListening(true);
    } else {
      recognitionRef.current.stop();
      setIsListening(false);
    }
  };

  const runPrediction = async () => {
    if (!smokingModel || !drinkingModel) {
      console.error("Models not loaded yet");
      return;
    }

    setIsPredicting(true);
    try {
      const updatedMetrics = { ...metrics };

      // Prepare features for smoking model
      const smokingFeatures = {
        BMI: updatedMetrics.BMI,
        SBP: updatedMetrics.SBP,
        DBP: updatedMetrics.DBP,
        BLDS: updatedMetrics.BLDS,
        gamma_GTP: updatedMetrics.gamma_GTP,
        triglyceride: updatedMetrics.triglyceride,
        DRK_YN: updatedMetrics.DRK_YN,
        hemoglobin: updatedMetrics.hemoglobin,
        waistline: updatedMetrics.waistline,
        HDL_chole: updatedMetrics.HDL_chole,
        sex_numeric: updatedMetrics.sex_numeric,
        LDL_chole: updatedMetrics.LDL_chole
      };

      // Prepare features for drinking model
      const drinkingFeatures = {
        BMI: updatedMetrics.BMI,
        SBP: updatedMetrics.SBP,
        DBP: updatedMetrics.DBP,
        BLDS: updatedMetrics.BLDS,
        gamma_GTP: updatedMetrics.gamma_GTP,
        triglyceride: updatedMetrics.triglyceride,
        SMK_stat_type_cd: updatedMetrics.SMK_stat_type_cd,
        hemoglobin: updatedMetrics.hemoglobin,
        waistline: updatedMetrics.waistline,
        HDL_chole: updatedMetrics.HDL_chole,
        sex_numeric: updatedMetrics.sex_numeric,
        LDL_chole: updatedMetrics.LDL_chole,
        urine_protein: updatedMetrics.urine_protein
      };

      const smokingProb = smokingModel.predict(smokingFeatures);
      const drinkingProb = drinkingModel.predict(drinkingFeatures);

      const insights = await getHealthInsights(updatedMetrics, smokingProb, drinkingProb);

      setPredictionResult({
        smokingProb,
        drinkingProb,
        insights: insights || "No insights available."
      });
    } catch (error) {
      console.error(error);
    } finally {
      setIsPredicting(false);
    }
  };

  const handleTabChange = (tab: 'home' | 'predict' | 'chat') => {
    if (tab === 'predict' && !user) {
      setAuthMode('signin');
      setIsAuthModalOpen(true);
      return;
    }
    setActiveTab(tab);
  };

  return (
    <ErrorBoundary>
      <div className="min-h-screen relative">
        <div className="atmosphere" />
        
        <AuthModal 
          isOpen={isAuthModalOpen}
          onClose={() => setIsAuthModalOpen(false)}
          mode={authMode}
          onModeChange={setAuthMode}
          onGoogleSignIn={handleGoogleSignIn}
          onAppleSignIn={handleAppleSignIn}
        />

        {/* Main Navigation Bar */}
        <nav className="fixed top-0 left-0 right-0 z-50 px-4 sm:px-6 py-4 flex justify-between items-center">
          {/* Logo & Name */}
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 sm:w-10 sm:h-10 rounded-xl bg-orange-500 flex items-center justify-center shadow-lg shadow-orange-500/20">
              <Activity className="text-white" size={20} />
            </div>
            <span className="text-base sm:text-lg lg:text-xl font-bold tracking-tight text-white">
              Smoke&DrinkRisk.AI
            </span>
          </div>

          {/* Desktop/Tablet Navigation */}
          <div className="hidden md:flex glass-card px-2 py-1.5 gap-1 items-center">
            {[
              { id: 'home', icon: Activity, label: 'Home' },
              { id: 'predict', icon: ShieldAlert, label: 'Predict' },
              { id: 'chat', icon: MessageSquare, label: 'Elena Chat' }
            ].map((tab) => (
              <button
                key={tab.id}
                id={`nav-${tab.id}`}
                onClick={() => handleTabChange(tab.id as any)}
                className={`flex items-center gap-2 px-4 py-2 rounded-xl transition-all ${
                  activeTab === tab.id 
                  ? 'bg-orange-500 text-white shadow-lg shadow-orange-500/20' 
                  : 'text-white/60 hover:text-white hover:bg-white/5'
                }`}
              >
                <tab.icon size={18} />
                <span className="font-medium">{tab.label}</span>
              </button>
            ))}
            
            <div className="w-px h-6 bg-white/10 mx-2" />
            
            <div className="relative">
              <button 
                onClick={() => setIsAuthMenuOpen(!isAuthMenuOpen)}
                className={`flex items-center gap-2 px-3 py-2 rounded-xl transition-all ${
                  isAuthMenuOpen ? 'bg-white/10 text-white' : 'text-white/60 hover:text-white hover:bg-white/5'
                }`}
              >
                {user ? (
                  <img 
                    src={user.photoURL || `https://ui-avatars.com/api/?name=${user.displayName}`} 
                    alt={user.displayName || 'User'} 
                    className="w-8 h-8 rounded-full border border-white/10"
                  />
                ) : (
                  <div className="w-8 h-8 rounded-full bg-white/5 flex items-center justify-center">
                    <UserIcon size={18} />
                  </div>
                )}
                <ChevronRight size={14} className={`transition-transform ${isAuthMenuOpen ? 'rotate-90' : ''}`} />
              </button>

              <AnimatePresence>
                {isAuthMenuOpen && (
                  <motion.div
                    initial={{ opacity: 0, y: 10, scale: 0.95 }}
                    animate={{ opacity: 1, y: 0, scale: 1 }}
                    exit={{ opacity: 0, y: 10, scale: 0.95 }}
                    className="absolute right-0 mt-2 w-64 glass-card p-2 shadow-2xl z-[60]"
                  >
                    {user ? (
                      <div className="space-y-1">
                        <div className="px-4 py-3 border-b border-white/5 mb-1">
                          <p className="font-bold truncate">{user.displayName}</p>
                          <p className="text-xs text-white/40 truncate">{user.email}</p>
                        </div>
                        <button 
                          onClick={() => {
                            handleTabChange('predict');
                            setIsAuthMenuOpen(false);
                          }}
                          className="w-full flex items-center gap-3 px-4 py-2.5 rounded-lg text-white/60 hover:text-white hover:bg-white/5 transition-all text-sm"
                        >
                          <Activity size={16} />
                          My Assessments
                        </button>
                        <button 
                          onClick={() => {
                            handleSignOut();
                            setIsAuthMenuOpen(false);
                          }}
                          className="w-full flex items-center gap-3 px-4 py-2.5 rounded-lg text-red-400 hover:text-red-300 hover:bg-red-500/10 transition-all text-sm"
                        >
                          <LogIn size={16} className="rotate-180" />
                          Sign Out
                        </button>
                      </div>
                    ) : (
                      <div className="space-y-1">
                        <div className="px-4 py-3 border-b border-white/5 mb-1">
                          <p className="font-bold">Welcome</p>
                          <p className="text-xs text-white/40">Sign in to sync your data</p>
                        </div>
                        <button 
                          onClick={() => {
                            handleSignIn();
                            setIsAuthMenuOpen(false);
                          }}
                          className="w-full flex items-center gap-3 px-4 py-2.5 rounded-lg text-white/60 hover:text-white hover:bg-white/5 transition-all text-sm"
                        >
                          <LogIn size={16} />
                          Sign In
                        </button>
                        <button 
                          onClick={() => {
                            handleSignUp();
                            setIsAuthMenuOpen(false);
                          }}
                          className="w-full flex items-center gap-3 px-4 py-2.5 rounded-lg bg-orange-500 text-white hover:bg-orange-600 transition-all text-sm font-semibold"
                        >
                          <UserPlus size={16} />
                          Sign Up
                        </button>
                      </div>
                    )}
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          </div>
        </nav>

        {/* Mobile Bottom Navigation */}
        <nav className="fixed bottom-0 left-0 right-0 z-50 px-4 py-3 md:hidden">
          <div className="glass-card px-2 py-1.5 flex justify-around items-center">
            {[
              { id: 'home', icon: Activity, label: 'Home' },
              { id: 'predict', icon: ShieldAlert, label: 'Predict' },
              { id: 'chat', icon: MessageSquare, label: 'Elena Chat' }
            ].map((tab) => (
              <button
                key={tab.id}
                onClick={() => handleTabChange(tab.id as any)}
                className={`flex flex-col items-center gap-1 px-3 py-1 rounded-xl transition-all ${
                  activeTab === tab.id 
                  ? 'text-orange-500' 
                  : 'text-white/40 hover:text-white'
                }`}
              >
                <tab.icon size={20} />
                <span className="text-[10px] font-medium">{tab.label}</span>
              </button>
            ))}
            
            <button 
              onClick={() => setIsAuthMenuOpen(!isAuthMenuOpen)}
              className={`flex flex-col items-center gap-1 px-3 py-1 rounded-xl transition-all ${
                isAuthMenuOpen ? 'text-orange-500' : 'text-white/40'
              }`}
            >
              {user ? (
                <img 
                  src={user.photoURL || `https://ui-avatars.com/api/?name=${user.displayName}`} 
                  alt={user.displayName || 'User'} 
                  className="w-5 h-5 rounded-full border border-white/10"
                />
              ) : (
                <UserIcon size={20} />
              )}
              <span className="text-[10px] font-medium">Profile</span>
            </button>
          </div>
        </nav>

        {/* Mobile Auth Menu Overlay */}
        <AnimatePresence>
          {isAuthMenuOpen && (
            <motion.div
              initial={{ opacity: 0, y: 100 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: 100 }}
              className="fixed inset-x-0 bottom-20 z-[60] px-4 md:hidden"
            >
              <div className="glass-card p-4 shadow-2xl space-y-4">
                {user ? (
                  <div className="space-y-3">
                    <div className="flex items-center gap-3 pb-3 border-b border-white/5">
                      <img 
                        src={user.photoURL || `https://ui-avatars.com/api/?name=${user.displayName}`} 
                        alt={user.displayName || 'User'} 
                        className="w-10 h-10 rounded-full border border-white/10"
                      />
                      <div>
                        <p className="font-bold">{user.displayName}</p>
                        <p className="text-xs text-white/40">{user.email}</p>
                      </div>
                    </div>
                    <button 
                      onClick={() => {
                        handleTabChange('predict');
                        setIsAuthMenuOpen(false);
                      }}
                      className="w-full flex items-center gap-3 px-4 py-3 rounded-xl bg-white/5 text-white/80"
                    >
                      <Activity size={18} />
                      My Assessments
                    </button>
                    <button 
                      onClick={() => {
                        handleSignOut();
                        setIsAuthMenuOpen(false);
                      }}
                      className="w-full flex items-center gap-3 px-4 py-3 rounded-xl bg-red-500/10 text-red-400"
                    >
                      <LogIn size={18} className="rotate-180" />
                      Sign Out
                    </button>
                  </div>
                ) : (
                  <div className="space-y-3">
                    <div className="pb-3 border-b border-white/5">
                      <p className="font-bold">Welcome</p>
                      <p className="text-xs text-white/40">Sign in to sync your data</p>
                    </div>
                    <button 
                      onClick={() => {
                        handleSignIn();
                        setIsAuthMenuOpen(false);
                      }}
                      className="w-full flex items-center gap-3 px-4 py-3 rounded-xl bg-white/5 text-white/80"
                    >
                      <LogIn size={18} />
                      Sign In
                    </button>
                    <button 
                      onClick={() => {
                        handleSignUp();
                        setIsAuthMenuOpen(false);
                      }}
                      className="w-full flex items-center gap-3 px-4 py-3 rounded-xl bg-orange-500 text-white font-bold"
                    >
                      <UserPlus size={18} />
                      Sign Up
                    </button>
                  </div>
                )}
                <button 
                  onClick={() => setIsAuthMenuOpen(false)}
                  className="w-full py-2 text-white/40 text-xs uppercase tracking-widest"
                >
                  Close
                </button>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

      <main className="pt-20 md:pt-24 pb-24 md:pb-12 px-4 sm:px-6">
        <AnimatePresence mode="wait">
          {activeTab === 'home' && (
            <motion.div
              key="home"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="space-y-8 md:space-y-12 max-w-5xl mx-auto"
            >
              <section className="text-center space-y-4 md:space-y-6 py-8 md:py-12">
                <motion.div
                  initial={{ scale: 0.9 }}
                  animate={{ scale: 1 }}
                  className="inline-flex items-center gap-2 px-3 py-1 md:px-4 md:py-1.5 rounded-full bg-orange-500/10 border border-orange-500/20 text-orange-400 text-xs md:text-sm font-medium"
                >
                  <Heart size={14} />
                  <span>AI-Powered Health Monitoring</span>
                </motion.div>
                <h1 className="text-4xl sm:text-5xl md:text-6xl font-bold tracking-tight leading-tight">
                  Your Journey to <br />
                  <span className="text-orange-500">Vitality</span> Starts Here
                </h1>
                <p className="text-lg md:text-xl text-white/60 max-w-2xl mx-auto">
                  Advanced health risk prediction using XGBoost machine learning and Gemini AI insights. 
                  Monitor smoking and drinking impacts with precision.
                </p>
                <div className="flex flex-col sm:flex-row justify-center gap-3 md:gap-4 pt-4">
                  <button 
                    onClick={() => handleTabChange('predict')}
                    className="px-6 py-3 md:px-8 md:py-4 bg-orange-500 hover:bg-orange-600 text-white rounded-xl md:rounded-2xl font-semibold transition-all flex items-center justify-center gap-2 group"
                  >
                    Start Prediction
                    <ChevronRight size={20} className="group-hover:translate-x-1 transition-transform" />
                  </button>
                  <button 
                    onClick={() => setActiveTab('chat')}
                    className="px-6 py-3 md:px-8 md:py-4 glass-card hover:bg-white/10 text-white rounded-xl md:rounded-2xl font-semibold transition-all"
                  >
                    Talk to Elena
                  </button>
                </div>
              </section>

              <div className="grid md:grid-cols-3 gap-6">
                {[
                  { title: 'Smart Prediction', desc: 'XGBoost models trained on clinical data to predict behavioral impacts.', icon: Activity },
                  { title: 'AI Insights', desc: 'Gemini-powered personalized health advice based on your unique metrics.', icon: Info },
                  { title: 'Voice Interaction', desc: 'Hands-free health consultation with real-time speech recognition.', icon: Mic }
                ].map((feature, i) => (
                  <div key={i} className="glass-card p-8 space-y-4 hover:border-orange-500/30 transition-colors">
                    <div className="w-12 h-12 rounded-xl bg-orange-500/10 flex items-center justify-center text-orange-500">
                      <feature.icon size={24} />
                    </div>
                    <h3 className="text-xl font-bold">{feature.title}</h3>
                    <p className="text-white/60">{feature.desc}</p>
                  </div>
                ))}
              </div>

              {/* Model Performance Section */}
              <section className="space-y-8">
                <div className="flex items-center gap-3 border-b border-white/10 pb-4">
                  <BarChart3 className="text-orange-500" />
                  <h2 className="text-3xl font-bold">Model Quality & Performance</h2>
                </div>
                <div className="grid md:grid-cols-2 gap-8">
                  <div className="glass-card p-8 space-y-6">
                    <h3 className="text-xl font-bold flex items-center gap-2">
                      <TrendingUp size={20} className="text-orange-500" />
                      Accuracy Metrics
                    </h3>
                    <div className="h-[300px] w-full">
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={performanceData}>
                          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                          <XAxis dataKey="name" stroke="rgba(255,255,255,0.4)" />
                          <YAxis stroke="rgba(255,255,255,0.4)" />
                          <Tooltip 
                            contentStyle={{ backgroundColor: 'rgba(0,0,0,0.8)', border: '1px solid rgba(255,255,255,0.1)' }}
                            itemStyle={{ color: '#f97316' }}
                          />
                          <Bar dataKey="value" fill="#f97316" radius={[4, 4, 0, 0]} />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                    <p className="text-white/60 text-sm">
                      Our XGBoost models are trained on over 100,000 clinical records, achieving high precision in predicting health risks associated with smoking and drinking.
                    </p>
                  </div>
                  <div className="glass-card p-8 space-y-6">
                    <h3 className="text-xl font-bold flex items-center gap-2">
                      <Activity size={20} className="text-blue-500" />
                      Model Distribution
                    </h3>
                    <div className="h-[300px] w-full flex items-center justify-center">
                      <ResponsiveContainer width="100%" height="100%">
                        <PieChart>
                          <Pie
                            data={modelDistribution}
                            cx="50%"
                            cy="50%"
                            innerRadius={60}
                            outerRadius={80}
                            paddingAngle={5}
                            dataKey="value"
                          >
                            {modelDistribution.map((entry, index) => (
                              <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                            ))}
                          </Pie>
                          <Tooltip 
                            contentStyle={{ backgroundColor: 'rgba(0,0,0,0.8)', border: '1px solid rgba(255,255,255,0.1)' }}
                          />
                        </PieChart>
                      </ResponsiveContainer>
                    </div>
                    <div className="grid grid-cols-2 gap-4">
                      <div className="flex items-center gap-2">
                        <div className="w-3 h-3 rounded-full bg-orange-500" />
                        <span className="text-sm text-white/60">Smoking Model</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <div className="w-3 h-3 rounded-full bg-blue-500" />
                        <span className="text-sm text-white/60">Drinking Model</span>
                      </div>
                    </div>
                  </div>
                </div>
              </section>

              {/* Recommended Section */}
              <section className="space-y-8">
                <div className="flex items-center gap-3 border-b border-white/10 pb-4">
                  <Star className="text-orange-500" />
                  <h2 className="text-3xl font-bold">Recommended for You</h2>
                </div>
                <div className="grid md:grid-cols-2 gap-8">
                  <div className="relative group overflow-hidden rounded-3xl">
                    <img 
                      src="https://picsum.photos/seed/doctor-patient/800/600" 
                      alt="Doctor consultation" 
                      className="w-full h-[300px] object-cover transition-transform group-hover:scale-105"
                      referrerPolicy="no-referrer"
                    />
                    <div className="absolute inset-0 bg-gradient-to-t from-black/80 to-transparent flex flex-col justify-end p-8">
                      <h3 className="text-2xl font-bold">Personalized Consultation</h3>
                      <p className="text-white/60">Connect with specialists who understand your clinical data.</p>
                    </div>
                  </div>
                  <div className="relative group overflow-hidden rounded-3xl">
                    <img 
                      src="https://picsum.photos/seed/healthy-living/800/600" 
                      alt="Healthy lifestyle" 
                      className="w-full h-[300px] object-cover transition-transform group-hover:scale-105"
                      referrerPolicy="no-referrer"
                    />
                    <div className="absolute inset-0 bg-gradient-to-t from-black/80 to-transparent flex flex-col justify-end p-8">
                      <h3 className="text-2xl font-bold">Lifestyle Habits</h3>
                      <p className="text-white/60">Explore curated programs to help you quit smoking and reduce drinking.</p>
                    </div>
                  </div>
                </div>
              </section>

              {/* Motivation & Advice Section */}
              <section className="space-y-8 py-12">
                <div className="flex items-center gap-3 border-b border-white/10 pb-4">
                  <Heart className="text-orange-500" />
                  <h2 className="text-3xl font-bold">Break Free, Live Better</h2>
                </div>
                
                <div className="grid md:grid-cols-2 gap-8">
                  <div className="glass-card p-8 space-y-6 border-orange-500/10">
                    <div className="w-14 h-14 rounded-2xl bg-orange-500/10 flex items-center justify-center text-orange-500">
                      <ShieldAlert size={28} />
                    </div>
                    <h3 className="text-2xl font-bold">Why Quit Smoking?</h3>
                    <ul className="space-y-4 text-white/70">
                      <li className="flex gap-3">
                        <div className="w-1.5 h-1.5 rounded-full bg-orange-500 mt-2 shrink-0" />
                        <p><span className="text-white font-semibold">Immediate Recovery:</span> Within 20 minutes, your heart rate drops. Within 48 hours, your sense of taste and smell improve.</p>
                      </li>
                      <li className="flex gap-3">
                        <div className="w-1.5 h-1.5 rounded-full bg-orange-500 mt-2 shrink-0" />
                        <p><span className="text-white font-semibold">Breathe Easier:</span> Your lung capacity increases by up to 30% within 2-3 months of quitting.</p>
                      </li>
                      <li className="flex gap-3">
                        <div className="w-1.5 h-1.5 rounded-full bg-orange-500 mt-2 shrink-0" />
                        <p><span className="text-white font-semibold">Long-term Health:</span> Your risk of heart disease drops by 50% after just one year of being smoke-free.</p>
                      </li>
                    </ul>
                  </div>

                  <div className="glass-card p-8 space-y-6 border-blue-500/10">
                    <div className="w-14 h-14 rounded-2xl bg-blue-500/10 flex items-center justify-center text-blue-500">
                      <Activity size={28} />
                    </div>
                    <h3 className="text-2xl font-bold">Benefits of Reducing Alcohol</h3>
                    <ul className="space-y-4 text-white/70">
                      <li className="flex gap-3">
                        <div className="w-1.5 h-1.5 rounded-full bg-blue-500 mt-2 shrink-0" />
                        <p><span className="text-white font-semibold">Better Sleep:</span> Alcohol disrupts REM sleep. Quitting leads to deeper, more restorative rest and higher energy levels.</p>
                      </li>
                      <li className="flex gap-3">
                        <div className="w-1.5 h-1.5 rounded-full bg-blue-500 mt-2 shrink-0" />
                        <p><span className="text-white font-semibold">Mental Wellness:</span> Reducing alcohol intake significantly lowers anxiety and depression symptoms over time.</p>
                      </li>
                      <li className="flex gap-3">
                        <div className="w-1.5 h-1.5 rounded-full bg-blue-500 mt-2 shrink-0" />
                        <p><span className="text-white font-semibold">Liver Health:</span> Your liver can begin to repair itself and shed excess fat within weeks of stopping alcohol consumption.</p>
                      </li>
                    </ul>
                  </div>
                </div>

                <div className="glass-card p-8 md:p-12 text-center space-y-6 bg-gradient-to-br from-orange-500/5 to-blue-500/5">
                  <h3 className="text-2xl md:text-3xl font-bold italic">"The secret of getting ahead is getting started."</h3>
                  <p className="text-white/60 max-w-2xl mx-auto leading-relaxed">
                    Every day is a new opportunity to choose health. You have the strength to overcome these habits. 
                    Use our AI Chat for daily motivation, track your progress in the Predict tab, and remember that 
                    a healthier version of you is waiting just around the corner.
                  </p>
                  <div className="pt-4">
                    <button 
                      onClick={() => setActiveTab('chat')}
                      className="px-8 py-4 bg-white text-black rounded-2xl font-bold hover:bg-white/90 transition-all shadow-xl"
                    >
                      Get Daily Motivation
                    </button>
                  </div>
                </div>
              </section>

              {/* Doctor & Patient Images Section */}
              <section className="space-y-8">
                <div className="flex items-center gap-3 border-b border-white/10 pb-4">
                  <Stethoscope className="text-orange-500" />
                  <h2 className="text-3xl font-bold">Clinical Support</h2>
                </div>
                <ClinicalSlider />
              </section>
            </motion.div>
          )}

          {activeTab === 'predict' && user && (
            <motion.div
              key="predict"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
              className="grid lg:grid-cols-2 gap-8 max-w-5xl mx-auto"
            >
              <div className="space-y-6">
                <div className="glass-card p-8 space-y-6">
                  <div className="flex items-center gap-3 border-b border-white/10 pb-4">
                    <Activity className="text-orange-500" />
                    <h2 className="text-2xl font-bold">Health Metrics</h2>
                  </div>
                  
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <label className="text-sm text-white/60">BMI</label>
                      <input 
                        type="number" 
                        step="0.1"
                        value={metrics.BMI}
                        onChange={(e) => setMetrics({...metrics, BMI: Number(e.target.value)})}
                        className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-3 focus:outline-none focus:border-orange-500/50"
                      />
                    </div>
                    <div className="space-y-2">
                      <label className="text-sm text-white/60">Hemoglobin (g/dL)</label>
                      <input 
                        type="number" 
                        step="0.1"
                        value={metrics.hemoglobin}
                        onChange={(e) => setMetrics({...metrics, hemoglobin: Number(e.target.value)})}
                        className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-3 focus:outline-none focus:border-orange-500/50"
                      />
                    </div>
                    <div className="space-y-2">
                      <label className="text-sm text-white/60">Sex</label>
                      <select 
                        value={metrics.sex_numeric}
                        onChange={(e) => setMetrics({...metrics, sex_numeric: Number(e.target.value)})}
                        className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-3 focus:outline-none focus:border-orange-500/50"
                      >
                        <option value={1}>Male</option>
                        <option value={2}>Female</option>
                      </select>
                    </div>
                    <div className="space-y-2">
                      <label className="text-sm text-white/60">SBP (mmHg)</label>
                      <input 
                        type="number" 
                        value={metrics.SBP}
                        onChange={(e) => setMetrics({...metrics, SBP: Number(e.target.value)})}
                        className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-3 focus:outline-none focus:border-orange-500/50"
                      />
                    </div>
                    <div className="space-y-2">
                      <label className="text-sm text-white/60">DBP (mmHg)</label>
                      <input 
                        type="number" 
                        value={metrics.DBP}
                        onChange={(e) => setMetrics({...metrics, DBP: Number(e.target.value)})}
                        className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-3 focus:outline-none focus:border-orange-500/50"
                      />
                    </div>
                    <div className="space-y-2">
                      <label className="text-sm text-white/60">Fasting Blood Sugar</label>
                      <input 
                        type="number" 
                        value={metrics.BLDS}
                        onChange={(e) => setMetrics({...metrics, BLDS: Number(e.target.value)})}
                        className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-3 focus:outline-none focus:border-orange-500/50"
                      />
                    </div>
                    <div className="space-y-2">
                      <label className="text-sm text-white/60">Gamma-GTP (U/L)</label>
                      <input 
                        type="number" 
                        value={metrics.gamma_GTP}
                        onChange={(e) => setMetrics({...metrics, gamma_GTP: Number(e.target.value)})}
                        className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-3 focus:outline-none focus:border-orange-500/50"
                      />
                    </div>
                    <div className="space-y-2">
                      <label className="text-sm text-white/60">Waistline (cm)</label>
                      <input 
                        type="number" 
                        value={metrics.waistline}
                        onChange={(e) => setMetrics({...metrics, waistline: Number(e.target.value)})}
                        className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-3 focus:outline-none focus:border-orange-500/50"
                      />
                    </div>
                    <div className="space-y-2">
                      <label className="text-sm text-white/60">HDL Cholesterol</label>
                      <input 
                        type="number" 
                        value={metrics.HDL_chole}
                        onChange={(e) => setMetrics({...metrics, HDL_chole: Number(e.target.value)})}
                        className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-3 focus:outline-none focus:border-orange-500/50"
                      />
                    </div>
                    <div className="space-y-2">
                      <label className="text-sm text-white/60">LDL Cholesterol</label>
                      <input 
                        type="number" 
                        value={metrics.LDL_chole}
                        onChange={(e) => setMetrics({...metrics, LDL_chole: Number(e.target.value)})}
                        className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-3 focus:outline-none focus:border-orange-500/50"
                      />
                    </div>
                    <div className="space-y-2">
                      <label className="text-sm text-white/60">Triglyceride</label>
                      <input 
                        type="number" 
                        value={metrics.triglyceride}
                        onChange={(e) => setMetrics({...metrics, triglyceride: Number(e.target.value)})}
                        className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-3 focus:outline-none focus:border-orange-500/50"
                      />
                    </div>
                    <div className="space-y-2">
                      <label className="text-sm text-white/60">Urine Protein</label>
                      <select 
                        value={metrics.urine_protein}
                        onChange={(e) => setMetrics({...metrics, urine_protein: Number(e.target.value)})}
                        className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-3 focus:outline-none focus:border-orange-500/50"
                      >
                        <option value={1}>Negative</option>
                        <option value={2}>Trace</option>
                        <option value={3}>1+</option>
                        <option value={4}>2+</option>
                        <option value={5}>3+</option>
                        <option value={6}>4+</option>
                      </select>
                    </div>
                  </div>

                  <div className="space-y-4 pt-4">
                    <h3 className="text-lg font-semibold">Behavioral Status</h3>
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                      <div className="space-y-2">
                        <label className="text-sm text-white/60">Smoking Status</label>
                        <select 
                          value={metrics.SMK_stat_type_cd}
                          onChange={(e) => setMetrics({...metrics, SMK_stat_type_cd: Number(e.target.value)})}
                          className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-3 focus:outline-none focus:border-orange-500/50"
                        >
                          <option value={1}>Never</option>
                          <option value={2}>Ex-smoker</option>
                          <option value={3}>Current Smoker</option>
                        </select>
                      </div>
                      <div className="space-y-2">
                        <label className="text-sm text-white/60">Drinking Status</label>
                        <select 
                          value={metrics.DRK_YN}
                          onChange={(e) => setMetrics({...metrics, DRK_YN: Number(e.target.value)})}
                          className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-3 focus:outline-none focus:border-orange-500/50"
                        >
                          <option value={0}>No</option>
                          <option value={1}>Yes</option>
                        </select>
                      </div>
                    </div>
                  </div>

                  <button 
                    onClick={runPrediction}
                    disabled={isPredicting || !smokingModel}
                    className="w-full py-4 bg-orange-500 hover:bg-orange-600 disabled:opacity-50 disabled:cursor-not-allowed text-white rounded-2xl font-bold text-lg transition-all shadow-lg shadow-orange-500/20"
                  >
                    {isPredicting ? 'Analyzing Data...' : 'Generate AI Assessment'}
                  </button>
                </div>
              </div>

              <div className="space-y-6">
                {!predictionResult ? (
                  <div className="glass-card p-12 flex flex-col items-center justify-center text-center space-y-6 h-full min-h-[400px]">
                    <div className="w-20 h-20 rounded-full bg-white/5 flex items-center justify-center text-white/20">
                      <Activity size={40} />
                    </div>
                    <div className="space-y-2">
                      <h3 className="text-2xl font-bold">Awaiting Analysis</h3>
                      <p className="text-white/40">Enter your medical metrics and behavioral status to receive a personalized health risk assessment.</p>
                    </div>
                  </div>
                ) : (
                  <motion.div
                    initial={{ opacity: 0, scale: 0.95 }}
                    animate={{ opacity: 1, scale: 1 }}
                    className="space-y-6"
                  >
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                      <div className="glass-card p-6 space-y-3">
                        <div className="flex items-center gap-2 text-orange-400">
                          <Wind size={18} />
                          <span className="text-sm font-medium uppercase tracking-wider">Smoking Risk</span>
                        </div>
                        <div className="text-3xl sm:text-4xl font-bold">{(predictionResult.smokingProb * 100).toFixed(1)}%</div>
                        <div className="w-full bg-white/10 h-2 rounded-full overflow-hidden">
                          <div 
                            className="h-full bg-orange-500 transition-all duration-1000" 
                            style={{ width: `${predictionResult.smokingProb * 100}%` }}
                          />
                        </div>
                      </div>
                      <div className="glass-card p-6 space-y-3">
                        <div className="flex items-center gap-2 text-blue-400">
                          <Droplets size={18} />
                          <span className="text-sm font-medium uppercase tracking-wider">Drinking Risk</span>
                        </div>
                        <div className="text-3xl sm:text-4xl font-bold">{(predictionResult.drinkingProb * 100).toFixed(1)}%</div>
                        <div className="w-full bg-white/10 h-2 rounded-full overflow-hidden">
                          <div 
                            className="h-full bg-blue-500 transition-all duration-1000" 
                            style={{ width: `${predictionResult.drinkingProb * 100}%` }}
                          />
                        </div>
                      </div>
                    </div>

                    <div className="glass-card p-8 space-y-6">
                      <div className="flex items-center gap-3 border-b border-white/10 pb-4">
                        <BarChart3 className="text-orange-500" />
                        <h3 className="text-xl font-bold">Metrics Comparison</h3>
                      </div>
                      <div className="space-y-4">
                        {[
                          { label: 'BMI', value: metrics.BMI, min: 18.5, max: 25, unit: '' },
                          { label: 'Blood Pressure (SBP)', value: metrics.SBP, min: 90, max: 120, unit: 'mmHg' },
                          { label: 'Fasting Blood Sugar', value: metrics.BLDS, min: 70, max: 100, unit: 'mg/dL' },
                          { label: 'Hemoglobin', value: metrics.hemoglobin, min: 12, max: 17, unit: 'g/dL' },
                          { label: 'Triglycerides', value: metrics.triglyceride, min: 50, max: 150, unit: 'mg/dL' }
                        ].map((m) => {
                          const percentage = Math.min(Math.max(((m.value - (m.min * 0.5)) / (m.max * 1.5 - (m.min * 0.5))) * 100, 0), 100);
                          const isHealthy = m.value >= m.min && m.value <= m.max;
                          return (
                            <div key={m.label} className="space-y-2">
                              <div className="flex justify-between text-sm">
                                <span className="text-white/60">{m.label}</span>
                                <span className={`font-bold ${isHealthy ? 'text-green-400' : 'text-orange-400'}`}>
                                  {m.value.toFixed(1)} {m.unit}
                                </span>
                              </div>
                              <div className="relative h-2 bg-white/5 rounded-full overflow-hidden">
                                <div 
                                  className="absolute h-full bg-white/10" 
                                  style={{ left: `${(m.min / (m.max * 1.5)) * 100}%`, width: `${((m.max - m.min) / (m.max * 1.5)) * 100}%` }}
                                />
                                <motion.div 
                                  initial={{ width: 0 }}
                                  animate={{ width: `${percentage}%` }}
                                  className={`absolute h-full rounded-full ${isHealthy ? 'bg-green-500' : 'bg-orange-500'}`}
                                />
                              </div>
                              <div className="flex justify-between text-[10px] text-white/20">
                                <span>Low</span>
                                <span>Normal Range: {m.min}-{m.max}</span>
                                <span>High</span>
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    </div>

                    <div className="glass-card p-8 space-y-4">
                      <div className="flex items-center gap-3 border-b border-white/10 pb-4">
                        <Info className="text-orange-500" />
                        <h3 className="text-xl font-bold">AI Clinical Insights</h3>
                      </div>
                      <div className="markdown-body text-white/80 leading-relaxed">
                        <Markdown>{predictionResult.insights}</Markdown>
                      </div>
                    </div>

                    <div className="p-4 rounded-2xl bg-orange-500/10 border border-orange-500/20 flex gap-3 text-orange-400">
                      <AlertTriangle size={20} className="shrink-0" />
                      <p className="text-sm">
                        <strong>Disclaimer:</strong> This assessment is generated by AI for informational purposes only. It is not a medical diagnosis. Please consult a healthcare professional for clinical advice.
                      </p>
                    </div>
                  </motion.div>
                )}
              </div>
            </motion.div>
          )}

          {activeTab === 'chat' && (
            <motion.div
              key="chat"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="glass-card h-[calc(100vh-10rem)] md:h-[calc(100vh-12rem)] flex flex-col overflow-hidden max-w-7xl mx-auto"
            >
              <div className="px-4 sm:px-8 py-4 border-b border-white/10 flex items-center justify-between bg-white/5">
                <div className="flex items-center gap-3">
                  <div className="w-8 h-8 sm:w-10 sm:h-10 rounded-full bg-orange-500 flex items-center justify-center text-white">
                    <Activity size={18} />
                  </div>
                  <div>
                    <h2 className="font-bold text-sm sm:text-base">Elena</h2>
                    <div className="flex items-center gap-1.5">
                      <div className={`w-1.5 h-1.5 sm:w-2 sm:h-2 rounded-full ${isLiveMode ? 'bg-red-500 animate-ping' : 'bg-green-500 animate-pulse'}`} />
                      <span className="text-[10px] sm:text-xs text-white/40">{isLiveMode ? 'Live Voice Mode' : 'Online & Ready'}</span>
                    </div>
                  </div>
                </div>
              </div>

              <div className="flex-1 overflow-y-auto p-4 sm:p-8 space-y-6">
                {messages.map((msg, i) => (
                  <motion.div
                    key={i}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
                  >
                    <div className={`max-w-[80%] flex gap-3 ${msg.role === 'user' ? 'flex-row-reverse' : ''}`}>
                      <div className={`w-8 h-8 rounded-full shrink-0 flex items-center justify-center mt-1 ${
                        msg.role === 'user' ? 'bg-white/10' : 'bg-orange-500'
                      }`}>
                        {msg.role === 'user' ? <UserIcon size={16} /> : <Activity size={16} />}
                      </div>
                      <div className="flex flex-col gap-1">
                        <div className={`px-5 py-3 rounded-2xl ${
                          msg.role === 'user' 
                          ? 'bg-orange-500 text-white rounded-tr-none' 
                          : 'glass-card rounded-tl-none'
                        }`}>
                          <div className="markdown-body">
                            <Markdown>{msg.text}</Markdown>
                          </div>
                        </div>
                        <div className={`flex items-center gap-2 px-1 ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                          <span className="text-[10px] text-white/30 font-medium">
                            {msg.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                          </span>
                          {msg.role === 'user' && (
                            <div className="flex items-center">
                              {msg.status === 'sent' && <CheckCircle2 size={10} className="text-white/20" />}
                              {msg.status === 'delivered' && (
                                <div className="flex -space-x-1">
                                  <CheckCircle2 size={10} className="text-white/40" />
                                  <CheckCircle2 size={10} className="text-white/40" />
                                </div>
                              )}
                              {msg.status === 'read' && (
                                <div className="flex -space-x-1">
                                  <CheckCircle2 size={10} className="text-blue-400" />
                                  <CheckCircle2 size={10} className="text-blue-400" />
                                </div>
                              )}
                            </div>
                          )}
                        </div>
                      </div>
                    </div>
                  </motion.div>
                ))}
                {isTyping && (
                  <div className="flex justify-start">
                    <div className="flex gap-3">
                      <div className="w-8 h-8 rounded-full bg-orange-500 flex items-center justify-center">
                        <Activity size={16} />
                      </div>
                      <div className="glass-card px-5 py-3 rounded-2xl rounded-tl-none flex items-center gap-1.5">
                        <div className="w-1.5 h-1.5 bg-orange-500 rounded-full animate-bounce" />
                        <div className="w-1.5 h-1.5 bg-orange-500 rounded-full animate-bounce [animation-delay:0.2s]" />
                        <div className="w-1.5 h-1.5 bg-orange-500 rounded-full animate-bounce [animation-delay:0.4s]" />
                        <span className="text-xs text-white/40 ml-2 font-medium">Elena is typing...</span>
                      </div>
                    </div>
                  </div>
                )}
                <div ref={chatEndRef} />
              </div>

              <form onSubmit={handleSendMessage} className="p-4 sm:p-6 bg-white/5 border-t border-white/10 flex gap-2 sm:gap-3">
                <button
                  type="button"
                  onClick={toggleVoice}
                  className={`p-3 sm:p-4 rounded-xl transition-all ${
                    isLiveMode || isListening ? 'bg-red-500 text-white animate-pulse' : 'glass-card text-white/60 hover:text-white'
                  }`}
                >
                  {isLiveMode || isListening ? <MicOff size={18} /> : <Mic size={18} />}
                </button>
                <input
                  type="text"
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  placeholder="Ask Elena..."
                  className="flex-1 bg-white/5 border border-white/10 rounded-xl px-4 sm:px-6 py-3 sm:py-4 text-sm sm:text-base focus:outline-none focus:border-orange-500/50"
                />
                <button
                  type="submit"
                  disabled={!input.trim()}
                  className="p-3 sm:p-4 bg-orange-500 hover:bg-orange-600 disabled:opacity-50 text-white rounded-xl transition-all"
                >
                  <Send size={18} />
                </button>
              </form>
            </motion.div>
          )}
        </AnimatePresence>
      </main>

      {/* Footer */}
      <footer className="py-12 px-6 border-t border-white/5 text-center">
        <div className="flex items-center justify-center gap-2 text-white/20 mb-4">
          <Activity size={16} />
          <span className="font-bold tracking-widest uppercase text-xs">Elena - AI Health Companion</span>
        </div>
        <p className="text-white/40 text-sm max-w-md mx-auto">
          Empowering you with data-driven insights for a healthier lifestyle. 
          Built with XGBoost and Gemini AI.
        </p>
      </footer>
    </div>
    </ErrorBoundary>
  );
}
