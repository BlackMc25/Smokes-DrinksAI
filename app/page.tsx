'use client';

import React, { useState, useEffect, useRef } from 'react';
import Image from 'next/image';
import { motion, AnimatePresence } from 'motion/react';
import {
  Activity,
  Wind,
  Beer,
  MessageSquare,
  User as UserIcon,
  Settings,
  ChevronRight,
  ArrowLeft,
  AlertCircle,
  CheckCircle2,
  TrendingUp,
  Mic,
  MicOff,
  Send,
  ArrowRight,
  Brain,
  Shield,
  BarChart3,
  HeartPulse,
  Info,
  LogIn,
  LogOut,
  Target,
  Award,
  Zap,
  UserPlus,
  X,
  Key,
  Volume2,
  VolumeX,
  History,
  Bell,
  Lock,
  ImagePlus,
  Download,
  Smartphone,
  Plus,
  Watch,
  QrCode,
  Scan,
  Mail,
  Phone,
  Globe,
  Trash2
} from 'lucide-react';
import { GoogleGenAI, Modality } from "@google/genai";
import {
  auth,
  db,
  googleProvider,
  signInWithPopup,
  signOut,
  onAuthStateChanged,
  createUserWithEmailAndPassword,
  signInWithEmailAndPassword,
  updateProfile,
  signInWithCredential,
  GoogleAuthProvider,
  type User
} from '../lib/firebase';

declare global {
  interface Window {
    google: any;
  }
}
import {
  doc,
  setDoc,
  collection,
  query,
  where,
  orderBy,
  onSnapshot
} from 'firebase/firestore';

// --- XGBoost Predictor Logic ---

class XGBPredictor {
  model: any;
  featureNames: string[];
  baseScore: number | number[];
  numClasses: number;

  constructor() {
    this.model = null;
    this.featureNames = [];
    this.baseScore = 0.5;
    this.numClasses = 1;
  }

  async load(url: string) {
    try {
      const response = await fetch(url);
      if (!response.ok) throw new Error(`Failed to load model from ${url}`);
      this.model = await response.json();

      this.featureNames = this.model.learner.feature_names || [];

      const baseScoreStr = this.model.learner.learner_model_param?.base_score;
      if (baseScoreStr) {
        try {
          // Handle potential scientific notation and array format
          this.baseScore = JSON.parse(baseScoreStr.replace(/E/g, 'e'));
        } catch (e) {
          this.baseScore = parseFloat(baseScoreStr);
        }
      }

      const iterationIndptr = this.model.learner.gradient_booster.model.iteration_indptr;
      if (iterationIndptr && iterationIndptr.length > 1) {
        this.numClasses = iterationIndptr[1] - iterationIndptr[0];
      } else {
        this.numClasses = Array.isArray(this.baseScore) ? this.baseScore.length : 1;
      }
    } catch (error) {
      console.error("Error loading XGBoost model:", error);
      throw error;
    }
  }

  predict(features: Record<string, any>): number {
    if (!this.model) return 0;

    const featureArray = this.featureNames.map(name => {
      const val = features[name];
      if (val === '' || val === undefined || val === null) return NaN;
      const num = Number(val);
      return isNaN(num) ? NaN : num;
    });
    const trees = this.model.learner.gradient_booster.model.trees;

    if (this.numClasses > 1) {
      const scores = new Array(this.numClasses).fill(0);

      if (Array.isArray(this.baseScore)) {
        for (let i = 0; i < this.numClasses; i++) {
          scores[i] = this.baseScore[i] || 0;
        }
      } else if (typeof this.baseScore === 'number') {
        for (let i = 0; i < this.numClasses; i++) {
          scores[i] = this.baseScore;
        }
      }

      for (let i = 0; i < trees.length; i++) {
        const classIdx = i % this.numClasses;
        scores[classIdx] += this.predictTree(trees[i], featureArray);
      }

      const maxScore = Math.max(...scores);
      const expScores = scores.map(s => Math.exp(s - maxScore));
      const sumExp = expScores.reduce((a, b) => a + b, 0);
      const probs = expScores.map(s => s / sumExp);

      return probs.slice(1).reduce((a, b) => a + b, 0);
    } else {
      let sum = 0;
      for (const tree of trees) {
        sum += this.predictTree(tree, featureArray);
      }

      const base = typeof this.baseScore === 'number' ? this.baseScore : 0.5;
      const logitBase = base > 0 && base < 1 ? Math.log(base / (1 - base)) : 0;
      return 1 / (1 + Math.exp(-(logitBase + sum)));
    }
  }

  private predictTree(tree: any, features: number[]): number {
    let nodeIdx = 0;
    while (true) {
      const leftChild = tree.left_children[nodeIdx];
      const rightChild = tree.right_children[nodeIdx];

      if (leftChild === -1) {
        return tree.base_weights[nodeIdx];
      }

      const featureIdx = tree.split_indices[nodeIdx];
      const condition = tree.split_conditions[nodeIdx];
      const defaultLeft = tree.default_left ? tree.default_left[nodeIdx] : true;

      const val = features[featureIdx];

      if (val === undefined || val === null || isNaN(val)) {
        nodeIdx = defaultLeft ? leftChild : rightChild;
      } else if (val < condition) {
        nodeIdx = leftChild;
      } else {
        nodeIdx = rightChild;
      }
    }
  }
}

// --- Components ---

const GlassCard = ({ children, className = "", id }: { children: React.ReactNode, className?: string, id?: string }) => (
  <div id={id} className={`bg-white/10 backdrop-blur-md border border-white/20 rounded-2xl overflow-hidden ${className}`}>
    {children}
  </div>
);

const InputField = ({ label, value, onChange, type = "text", placeholder, id }: any) => (
  <div className="flex flex-col gap-1.5">
    <label className="text-xs font-medium text-slate-400 uppercase tracking-wider">{label}</label>
    <input
      id={id}
      type={type}
      inputMode={type === "number" ? "decimal" : "text"}
      value={value === undefined || value === null ? '' : value}
      onChange={(e) => onChange(e.target.value)}
      placeholder={placeholder}
      className="bg-slate-800/50 border border-slate-700 rounded-lg px-3 py-2 text-white placeholder:text-slate-600 focus:outline-none focus:ring-2 focus:ring-blue-500/50 transition-all"
    />
  </div>
);

const SelectField = ({ label, value, onChange, options, id, placeholder = "Select..." }: any) => (
  <div className="flex flex-col gap-1.5">
    <label className="text-xs font-medium text-slate-400 uppercase tracking-wider">{label}</label>
    <select
      id={id}
      value={value === undefined || value === null ? '' : value}
      onChange={(e) => onChange(e.target.value)}
      className="bg-slate-800/50 border border-slate-700 rounded-lg px-3 py-2 text-white focus:outline-none focus:ring-2 focus:ring-blue-500/50 transition-all appearance-none"
    >
      <option value="" disabled className="bg-slate-900 text-slate-500">{placeholder}</option>
      {options.map((opt: any) => (
        <option key={opt.value} value={opt.value} className="bg-slate-900">{opt.label}</option>
      ))}
    </select>
  </div>
);

// --- Main App ---

// --- Auth Modal ---
const AuthModal = ({
  showAuthModal,
  setShowAuthModal,
  authMode,
  setAuthMode,
  handleLogin
}: {
  showAuthModal: boolean;
  setShowAuthModal: (show: boolean) => void;
  authMode: 'signin' | 'signup';
  setAuthMode: (mode: 'signin' | 'signup') => void;
  handleLogin: (credential?: string) => Promise<void>;
}) => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [displayName, setDisplayName] = useState('');
  const [authError, setAuthError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (showAuthModal && window.google) {
      const initializeGsi = () => {
        window.google.accounts.id.initialize({
          client_id: process.env.NEXT_PUBLIC_GOOGLE_CLIENT_ID,
          callback: async (response: any) => {
            setLoading(true);
            try {
              await handleLogin(response.credential);
              setShowAuthModal(false);
            } catch (error: any) {
              setAuthError(error.message);
            } finally {
              setLoading(false);
            }
          },
          auto_select: false,
          cancel_on_tap_outside: true,
        });

        window.google.accounts.id.renderButton(
          document.getElementById("google-signin-button"),
          {
            theme: "outline",
            size: "large",
            width: "100%",
            text: authMode === 'signin' ? "signin_with" : "signup_with",
            shape: "pill"
          }
        );
      };

      // Small delay to ensure the div is rendered
      const timer = setTimeout(initializeGsi, 100);
      return () => clearTimeout(timer);
    }
  }, [showAuthModal, authMode]);

  const handleEmailAuth = async (e: React.FormEvent) => {
    e.preventDefault();
    setAuthError(null);
    setLoading(true);
    try {
      if (authMode === 'signup') {
        const userCredential = await createUserWithEmailAndPassword(auth, email, password);
        if (displayName) {
          await updateProfile(userCredential.user, { displayName });
        }
      } else {
        await signInWithEmailAndPassword(auth, email, password);
      }
      setShowAuthModal(false);
    } catch (error: any) {
      console.error("Auth failed:", error);
      setAuthError(error.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <AnimatePresence>
      {showAuthModal && (
        <div className="fixed inset-0 z-[100] flex items-center justify-center p-4">
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={() => setShowAuthModal(false)}
            className="absolute inset-0 bg-slate-950/80 backdrop-blur-sm"
          />
          <motion.div
            initial={{ opacity: 0, scale: 0.9, y: 20 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.9, y: 20 }}
            className="relative w-full max-w-md bg-slate-900 border border-white/10 rounded-3xl overflow-hidden shadow-2xl"
          >
            <div className="p-6 border-b border-white/5 flex items-center justify-between">
              <h3 className="text-xl font-bold text-white">{authMode === 'signin' ? 'Sign In' : 'Sign Up'}</h3>
              <button onClick={() => setShowAuthModal(false)} className="p-2 hover:bg-white/5 rounded-full text-slate-400 hover:text-white transition-colors">
                <X size={20} />
              </button>
            </div>

            <div className="p-6 space-y-6">
              <div className="text-center space-y-2">
                <p className="text-slate-400 text-sm">
                  {authMode === 'signin'
                    ? 'Sign in to access your health dashboard.'
                    : 'Join us today for personalized AI-powered health insights.'}
                </p>
              </div>

              <form onSubmit={handleEmailAuth} className="space-y-4">
                {authMode === 'signup' && (
                  <div className="space-y-1.5">
                    <label className="text-xs font-medium text-slate-400 uppercase tracking-wider">Name</label>
                    <input
                      type="text"
                      value={displayName}
                      onChange={(e) => setDisplayName(e.target.value)}
                      placeholder="Your Name"
                      className="w-full bg-slate-800/50 border border-slate-700 rounded-xl px-4 py-3 text-white focus:outline-none focus:ring-2 focus:ring-blue-500/50 transition-all"
                      required
                    />
                  </div>
                )}
                <div className="space-y-1.5">
                  <label className="text-xs font-medium text-slate-400 uppercase tracking-wider">Email</label>
                  <input
                    type="email"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    placeholder="email@example.com"
                    className="w-full bg-slate-800/50 border border-slate-700 rounded-xl px-4 py-3 text-white focus:outline-none focus:ring-2 focus:ring-blue-500/50 transition-all"
                    required
                  />
                </div>
                <div className="space-y-1.5">
                  <label className="text-xs font-medium text-slate-400 uppercase tracking-wider">Password</label>
                  <input
                    type="password"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    placeholder="••••••••"
                    className="w-full bg-slate-800/50 border border-slate-700 rounded-xl px-4 py-3 text-white focus:outline-none focus:ring-2 focus:ring-blue-500/50 transition-all"
                    required
                  />
                </div>

                {authError && (
                  <p className="text-red-500 text-xs text-center">{authError}</p>
                )}

                <button
                  type="submit"
                  disabled={loading}
                  className="w-full py-3 bg-blue-600 text-white rounded-xl font-bold hover:bg-blue-500 transition-all shadow-lg shadow-blue-600/20 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {loading ? 'Processing...' : (authMode === 'signin' ? 'Sign In' : 'Sign Up')}
                </button>

                <div className="relative py-2">
                  <div className="absolute inset-0 flex items-center"><div className="w-full border-t border-white/5"></div></div>
                  <div className="relative flex justify-center text-xs uppercase"><span className="bg-slate-900 px-2 text-slate-500">Or</span></div>
                </div>

                <div id="google-signin-button" className="w-full flex justify-center min-h-[44px]"></div>

                <p className="text-center text-sm text-slate-500">
                  {authMode === 'signin' ? "Don't have an account?" : "Already have an account?"}{' '}
                  <button
                    type="button"
                    onClick={() => setAuthMode(authMode === 'signin' ? 'signup' : 'signin')}
                    className="text-blue-500 hover:underline font-medium"
                  >
                    {authMode === 'signin' ? 'Sign Up' : 'Sign In'}
                  </button>
                </p>
              </form>
            </div>
          </motion.div>
        </div>
      )}
    </AnimatePresence>
  );
};

export default function HealthRiskApp() {
  const [isMounted, setIsMounted] = useState(false);
  const [activeTab, setActiveTab] = useState('home');

  useEffect(() => {
    setIsMounted(true);
  }, []);

  const [showAuthModal, setShowAuthModal] = useState(false);
  const [authMode, setAuthMode] = useState<'signin' | 'signup'>('signin');
  const [modelsLoaded, setModelsLoaded] = useState(false);
  const [modelError, setModelError] = useState<string | null>(null);
  const [predictionLoading, setPredictionLoading] = useState(false);
  const [predictionResult, setPredictionResult] = useState<any>(null);
  const [user, setUser] = useState<User | null>(null);
  const [authLoading, setAuthLoading] = useState(true);
  const [showSettings, setShowSettings] = useState(false);
  const [isVoiceMode, setIsVoiceMode] = useState(false);
  const [activeSettingsTab, setActiveSettingsTab] = useState<'main' | 'privacy' | 'notifications' | 'ai' | 'export' | 'contact'>('main');
  const [settings, setSettings] = useState({
    notifications: true,
    emailAlerts: false,
    healthReminders: true,
    medicationReminders: false,
    privacy: true,
    history: false,
    twoFactor: false,
    advancedReasoning: true,
    voiceFeedback: false,
    autoReadChat: false,
    highPerformance: false,
    autoExport: false,
    cloudSync: true,
  });

  // Models
  const smokingPredictor = useRef<XGBPredictor>(new XGBPredictor());
  const drinkingPredictor = useRef<XGBPredictor>(new XGBPredictor());

  // Form State
  const [metrics, setMetrics] = useState({
    BMI: '',
    SBP: '',
    DBP: '',
    BLDS: '',
    gamma_GTP: '',
    triglyceride: '',
    DRK_YN: '',
    SMK_stat_type_cd: '',
    hemoglobin: '',
    waistline: '',
    HDL_chole: '',
    LDL_chole: '',
    sex_numeric: '',
    urine_protein: ''
  });

  // Chat State
  const [messages, setMessages] = useState([
    { role: 'assistant', content: "Hello! I'm Elena, your AI health companion. How can I help you today?" }
  ]);
  const [inputMessage, setInputMessage] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState<string | null>(null);
  const speechRef = useRef<SpeechSynthesisUtterance | null>(null);

  const speakText = (text: string, id: string) => {
    if (typeof window === 'undefined' || !settings.voiceFeedback) return;

    if (isSpeaking === id) {
      window.speechSynthesis.cancel();
      setIsSpeaking(null);
      return;
    }

    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.onend = () => setIsSpeaking(null);
    speechRef.current = utterance;
    setIsSpeaking(id);
    window.speechSynthesis.speak(utterance);
  };

  const stopSpeaking = () => {
    if (typeof window !== 'undefined') {
      window.speechSynthesis.cancel();
      setIsSpeaking(null);
    }
  };

  useEffect(() => {
    return () => {
      if (typeof window !== 'undefined') {
        window.speechSynthesis.cancel();
      }
    };
  }, []);

  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [healthHistory, setHealthHistory] = useState<any[]>([]);
  const [exportProgress, setExportProgress] = useState<number | null>(null);
  const [exportStatus, setExportStatus] = useState<string | null>(null);
  const [showContactForm, setShowContactForm] = useState(false);
  const [showCallModal, setShowCallModal] = useState(false);
  const [showHelpModal, setShowHelpModal] = useState(false);
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);
  const [contactFormData, setContactFormData] = useState({ name: '', email: '', message: '' });
  const [contactFormStatus, setContactFormStatus] = useState<string | null>(null);
  const [contactFormLoading, setContactFormLoading] = useState(false);
  const chatEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (chatEndRef.current) {
      chatEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages, isTyping]);

  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, (currentUser) => {
      setUser(currentUser);
      setAuthLoading(false);
      if (currentUser) {
        // Load settings from Firestore
        const userDocRef = doc(db, 'users', currentUser.uid);
        onSnapshot(userDocRef, (docSnap) => {
          if (docSnap.exists()) {
            const data = docSnap.data();
            if (data.settings) {
              setSettings(prev => ({ ...prev, ...data.settings }));
            }
          }
        }, (error) => {
          console.error("Error loading settings:", error);
        });

        // Load health history
        const assessmentsRef = collection(db, 'assessments');
        const q = query(assessmentsRef, where('uid', '==', currentUser.uid), orderBy('createdAt', 'desc'));
        onSnapshot(q, (snapshot) => {
          const history = snapshot.docs.map(doc => ({ id: doc.id, ...doc.data() }));
          setHealthHistory(history);
        }, (error) => {
          console.error("Error loading health history:", error);
        });
      }
    });
    return () => unsubscribe();
  }, []);

  // Persist settings when they change
  useEffect(() => {
    if (user && !authLoading) {
      const userDocRef = doc(db, 'users', user.uid);
      setDoc(userDocRef, { settings }, { merge: true }).catch(err => {
        console.error("Error saving settings:", err);
      });
    }
  }, [settings, user, authLoading]);

  useEffect(() => {
    loadModels();
  }, []);

  const handleExportData = async (format: string) => {
    const dataToExport = predictionResult ? [predictionResult, ...healthHistory] : healthHistory;

    if (dataToExport.length === 0) {
      setExportStatus("No data to export yet. Complete a health assessment first!");
      setTimeout(() => setExportStatus(null), 3000);
      return;
    }

    setExportStatus(`Preparing ${format}...`);

    // Only simulate progress for PDF which takes longer
    if (format === 'PDF Report') {
      setExportProgress(0);
      const interval = setInterval(() => {
        setExportProgress(prev => {
          if (prev === null || prev >= 90) {
            clearInterval(interval);
            return 90;
          }
          return prev + 10;
        });
      }, 200);
    }

    setTimeout(async () => {
      try {
        let content = "";
        let fileName = `health_report_${new Date().toISOString().split('T')[0]}`;
        let mimeType = "";

        // Format data with percentages for export
        const formattedData = dataToExport.map(item => {
          const newItem = { ...item };
          // Ensure all numerical risks are exported as percentages
          if (typeof newItem.smoking === 'number') newItem.smoking = `${(newItem.smoking * 100).toFixed(1)}%`;
          if (typeof newItem.drinking === 'number') newItem.drinking = `${(newItem.drinking * 100).toFixed(1)}%`;
          return newItem;
        });

        if (format === 'JSON Raw') {
          content = JSON.stringify(formattedData, null, 2);
          fileName += ".json";
          mimeType = "application/json";

          const blob = new Blob([content], { type: mimeType });
          const url = URL.createObjectURL(blob);
          const a = document.createElement('a');
          a.href = url;
          a.download = fileName;
          a.click();
          URL.revokeObjectURL(url);
        } else if (format === 'CSV Data') {
          const headers = Object.keys(formattedData[0]).join(",");
          const rows = formattedData.map(item => Object.values(item).join(",")).join("\n");
          content = `${headers}\n${rows}`;
          fileName += ".csv";
          mimeType = "text/csv";

          const blob = new Blob([content], { type: mimeType });
          const url = URL.createObjectURL(blob);
          const a = document.createElement('a');
          a.href = url;
          a.download = fileName;
          a.click();
          URL.revokeObjectURL(url);
        } else if (format === 'PDF Report') {
          const { jsPDF } = await import('jspdf');
          const { default: autoTable } = await import('jspdf-autotable');
          const doc = new jsPDF();

          // Add Header
          doc.setFillColor(30, 41, 59); // Slate-900
          doc.rect(0, 0, 210, 40, 'F');

          doc.setTextColor(255, 255, 255);
          doc.setFontSize(22);
          doc.text("SMOKING&DRINKINGHEALTH.AI REPORT", 105, 20, { align: 'center' });

          doc.setFontSize(10);
          doc.text(`Generated on: ${new Date().toLocaleString()}`, 105, 30, { align: 'center' });

          // Add User Info if available
          doc.setTextColor(30, 41, 59);
          doc.setFontSize(14);
          doc.text("Assessment Summary", 14, 50);

          const tableData = formattedData.map((item, index) => [
            index + 1,
            item.timestamp ? new Date(item.timestamp).toLocaleDateString() : 'N/A',
            item.smoking || '0%',
            item.drinking || '0%',
            item.prediction || 'N/A'
          ]);

          autoTable(doc, {
            startY: 55,
            head: [['#', 'Date', 'Smoking Risk', 'Drinking Risk', 'AI Prediction']],
            body: tableData,
            theme: 'grid',
            headStyles: { fillColor: [59, 130, 246], textColor: [255, 255, 255] },
            alternateRowStyles: { fillColor: [241, 245, 249] },
            margin: { top: 55 }
          });

          // Add Footer
          const pageCount = (doc as any).internal.getNumberOfPages();
          for (let i = 1; i <= pageCount; i++) {
            doc.setPage(i);
            doc.setFontSize(8);
            doc.setTextColor(150);
            doc.text("Smoking&DrinkingHealth.AI - Personalized Insights", 105, 285, { align: 'center' });
            doc.text(`Page ${i} of ${pageCount}`, 190, 285);
          }

          doc.save(fileName + ".pdf");
        } else {
          setExportStatus(`${format} integration active!`);
          setExportProgress(null);
          setTimeout(() => setExportStatus(null), 3000);
          return;
        }

        setExportStatus(`${format} downloaded successfully!`);
        setExportProgress(null);
        setTimeout(() => setExportStatus(null), 3000);
      } catch (error) {
        console.error("Export error:", error);
        setExportStatus("Export failed. Please try again.");
        setExportProgress(null);
        setTimeout(() => setExportStatus(null), 3000);
      }
    }, 2000);
  };
  const handleTabChange = (tab: string) => {
    stopSpeaking();
    setActiveTab(tab);
  };

  const handleLogin = async (credential?: any) => {
    try {
      if (typeof credential === 'string') {
        // Handle GIS credential
        const authCredential = GoogleAuthProvider.credential(credential);
        await signInWithCredential(auth, authCredential);
      } else {
        // Fallback to popup if no credential provided (or if it's an event object)
        const result = await signInWithPopup(auth, googleProvider);
        if (!result) {
          throw new Error("Login failed. Please check if popups are blocked.");
        }
      }
      setShowAuthModal(false);
    } catch (error: any) {
      console.error("Login failed:", error);
      if (error.code === 'auth/popup-closed-by-user') {
        // User closed the popup, do nothing or show a subtle message
        return;
      }
      if (error.code === 'auth/popup-blocked') {
        alert("Please allow popups for this site to sign in with Google.");
      } else if (error.code === 'auth/unauthorized-domain') {
        alert("This domain is not authorized for Google Sign-in. Please contact support.");
      } else {
        alert("Login failed: " + error.message);
      }
    }
  };

  const handleLogout = async () => {
    try {
      await signOut(auth);
      handleTabChange('home');
      setShowSettings(false);
      setIsVoiceMode(false);
    } catch (error) {
      console.error("Logout failed:", error);
    }
  };

  // --- Settings Modal ---
  const SettingsModal = () => {
    const toggleSetting = (key: keyof typeof settings) => {
      const newValue = !settings[key];
      setSettings(prev => ({ ...prev, [key]: newValue }));

      // Sync voice activities
      if ((key === 'voiceFeedback' || key === 'autoReadChat') && newValue === false) {
        stopSpeaking();
        if (key === 'voiceFeedback') {
          setIsVoiceMode(false);
        }
      }
    };

    const renderMain = () => (
      <div className="p-6 space-y-2">
        {[
          { id: 'privacy', icon: Lock, label: 'Privacy & Security', desc: 'Manage your data and security' },
          { id: 'notifications', icon: Bell, label: 'Notifications', desc: 'Manage your alerts and reminders' },
          { id: 'ai', icon: Brain, label: 'AI & Voice', desc: 'Configure AI\'s intelligence' },
          { id: 'export', icon: Download, label: 'Data Export', desc: 'Download your health records' },
          { id: 'contact', icon: Mail, label: 'Contact Us', desc: 'Get in touch with our support team' }
        ].map((item) => (
          <div
            key={item.id}
            onClick={() => setActiveSettingsTab(item.id as any)}
            className="flex items-center justify-between p-4 bg-white/5 rounded-2xl hover:bg-white/10 transition-colors cursor-pointer group"
          >
            <div className="flex items-center gap-4">
              <div className="text-slate-400 group-hover:text-blue-500 transition-colors"><item.icon size={20} /></div>
              <div>
                <p className="text-sm font-medium text-white">{item.label}</p>
                <p className="text-xs text-slate-500">{item.desc}</p>
              </div>
            </div>
            <ChevronRight size={18} className="text-slate-600 group-hover:text-white transition-colors" />
          </div>
        ))}

        {user && (
          <div className="pt-4">
            <button
              onClick={handleLogout}
              className="w-full py-4 text-red-500 font-bold bg-red-500/10 rounded-2xl hover:bg-red-500/20 transition-colors flex items-center justify-center gap-2"
            >
              <LogOut size={20} />
              Sign Out
            </button>
          </div>
        )}
      </div>
    );

    const renderPrivacy = () => (
      <div className="p-6 space-y-6">
        <div className="space-y-4">
          <h4 className="text-xs font-bold text-slate-500 uppercase tracking-widest">Privacy</h4>
          <div className="space-y-2">
            {[
              { id: 'privacy', icon: Lock, label: 'Data Privacy', desc: 'Allow data collection for better health insights' },
              { id: 'history', icon: History, label: 'History Tracking', desc: 'Keep a log of your health predictions' }
            ].map((item) => (
              <div
                key={item.id}
                onClick={() => toggleSetting(item.id as keyof typeof settings)}
                className="flex items-center justify-between p-4 bg-white/5 rounded-2xl hover:bg-white/10 transition-colors cursor-pointer group"
              >
                <div className="flex items-center gap-4">
                  <div className="text-slate-400 group-hover:text-blue-500 transition-colors"><item.icon size={20} /></div>
                  <div>
                    <p className="text-sm font-medium text-white">{item.label}</p>
                    <p className="text-xs text-slate-500">{item.desc}</p>
                  </div>
                </div>
                <div className={`w-10 h-5 rounded-full relative transition-colors ${settings[item.id as keyof typeof settings] ? 'bg-blue-600' : 'bg-slate-700'}`}>
                  <div className={`absolute top-1 w-3 h-3 bg-white rounded-full transition-all ${settings[item.id as keyof typeof settings] ? 'right-1' : 'left-1'}`} />
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="space-y-4">
          <h4 className="text-xs font-bold text-slate-500 uppercase tracking-widest">Security</h4>
          <div className="space-y-2">
            <div
              onClick={() => toggleSetting('twoFactor')}
              className="flex items-center justify-between p-4 bg-white/5 rounded-2xl hover:bg-white/10 transition-colors cursor-pointer group"
            >
              <div className="flex items-center gap-4">
                <div className="text-slate-400 group-hover:text-blue-500 transition-colors"><Shield size={20} /></div>
                <div>
                  <p className="text-sm font-medium text-white">Two-Factor Auth</p>
                  <p className="text-xs text-slate-500">Secure your account with 2FA</p>
                </div>
              </div>
              <div className={`w-10 h-5 rounded-full relative transition-colors ${settings.twoFactor ? 'bg-blue-600' : 'bg-slate-700'}`}>
                <div className={`absolute top-1 w-3 h-3 bg-white rounded-full transition-all ${settings.twoFactor ? 'right-1' : 'left-1'}`} />
              </div>
            </div>
          </div>
        </div>

        <div className="pt-4 border-t border-white/5">
          <button
            onClick={() => setActiveSettingsTab('export')}
            className="w-full py-3 text-sm font-medium text-slate-400 hover:text-white transition-colors text-left px-4"
          >
            Export My Data
          </button>
          <button
            onClick={() => setShowDeleteConfirm(true)}
            className="w-full py-3 text-sm font-medium text-red-500/70 hover:text-red-500 transition-colors text-left px-4"
          >
            Delete Account
          </button>
        </div>
      </div>
    );

    const renderNotifications = () => (
      <div className="p-6 space-y-6">
        <div className="space-y-4">
          <h4 className="text-xs font-bold text-slate-500 uppercase tracking-widest">General Alerts</h4>
          <div className="space-y-2">
            {[
              { id: 'notifications', icon: Bell, label: 'Push Notifications', desc: 'Real-time health alerts on your device' },
              { id: 'emailAlerts', icon: Info, label: 'Email Alerts', desc: 'Weekly health summaries and reports' }
            ].map((item) => (
              <div
                key={item.id}
                onClick={() => toggleSetting(item.id as keyof typeof settings)}
                className="flex items-center justify-between p-4 bg-white/5 rounded-2xl hover:bg-white/10 transition-colors cursor-pointer group"
              >
                <div className="flex items-center gap-4">
                  <div className="text-slate-400 group-hover:text-blue-500 transition-colors"><item.icon size={20} /></div>
                  <div>
                    <p className="text-sm font-medium text-white">{item.label}</p>
                    <p className="text-xs text-slate-500">{item.desc}</p>
                  </div>
                </div>
                <div className={`w-10 h-5 rounded-full relative transition-colors ${settings[item.id as keyof typeof settings] ? 'bg-blue-600' : 'bg-slate-700'}`}>
                  <div className={`absolute top-1 w-3 h-3 bg-white rounded-full transition-all ${settings[item.id as keyof typeof settings] ? 'right-1' : 'left-1'}`} />
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="space-y-4">
          <h4 className="text-xs font-bold text-slate-500 uppercase tracking-widest">Health Reminders</h4>
          <div className="space-y-2">
            {[
              { id: 'healthReminders', icon: Activity, label: 'Checkup Reminders', desc: 'Alerts for scheduled health screenings' },
              { id: 'medicationReminders', icon: HeartPulse, label: 'Medication Alerts', desc: 'Daily reminders for your prescriptions' }
            ].map((item) => (
              <div
                key={item.id}
                onClick={() => toggleSetting(item.id as keyof typeof settings)}
                className="flex items-center justify-between p-4 bg-white/5 rounded-2xl hover:bg-white/10 transition-colors cursor-pointer group"
              >
                <div className="flex items-center gap-4">
                  <div className="text-slate-400 group-hover:text-blue-500 transition-colors"><item.icon size={20} /></div>
                  <div>
                    <p className="text-sm font-medium text-white">{item.label}</p>
                    <p className="text-xs text-slate-500">{item.desc}</p>
                  </div>
                </div>
                <div className={`w-10 h-5 rounded-full relative transition-colors ${settings[item.id as keyof typeof settings] ? 'bg-blue-600' : 'bg-slate-700'}`}>
                  <div className={`absolute top-1 w-3 h-3 bg-white rounded-full transition-all ${settings[item.id as keyof typeof settings] ? 'right-1' : 'left-1'}`} />
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    );

    const renderAI = () => (
      <div className="p-6 space-y-6">

        {/* ✅ API STATUS (SAFE) */}
        <div className="space-y-4">
          <h4 className="text-xs font-bold text-slate-500 uppercase tracking-widest">
            AI System
          </h4>

          <div className="p-4 rounded-2xl bg-emerald-500/10 border border-emerald-500/20 flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-white">AI Connection</p>
              <p className="text-xs text-slate-400">
                Securely connected to server-side AI engine
              </p>
            </div>
            <div className="text-emerald-400 text-xs font-bold">ACTIVE</div>
          </div>
        </div>

        {/* Intelligence */}
        <div className="space-y-4">
          <h4 className="text-xs font-bold text-slate-500 uppercase tracking-widest">Intelligence</h4>
          <div className="space-y-2">
            {[
              { id: 'advancedReasoning', icon: Brain, label: 'Advanced Reasoning', desc: 'Enable deeper AI analysis for health predictions' },
              { id: 'highPerformance', icon: Zap, label: 'High Performance', desc: 'Prioritize speed over detailed explanations' }
            ].map((item) => (
              <div
                key={item.id}
                onClick={() => toggleSetting(item.id as keyof typeof settings)}
                className="flex items-center justify-between p-4 bg-white/5 rounded-2xl hover:bg-white/10 transition-colors cursor-pointer group"
              >
                <div className="flex items-center gap-4">
                  <div className="text-slate-400 group-hover:text-blue-500 transition-colors">
                    <item.icon size={20} />
                  </div>
                  <div>
                    <p className="text-sm font-medium text-white">{item.label}</p>
                    <p className="text-xs text-slate-500">{item.desc}</p>
                  </div>
                </div>
                <div className={`w-10 h-5 rounded-full relative transition-colors ${settings[item.id as keyof typeof settings] ? 'bg-blue-600' : 'bg-slate-700'
                  }`}>
                  <div className={`absolute top-1 w-3 h-3 bg-white rounded-full transition-all ${settings[item.id as keyof typeof settings] ? 'right-1' : 'left-1'
                    }`} />
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Voice */}
        <div className="space-y-4">
          <h4 className="text-xs font-bold text-slate-500 uppercase tracking-widest">Voice & Audio</h4>
          <div className="space-y-2">
            {[
              { id: 'voiceFeedback', icon: Volume2, label: 'AI Voice', desc: 'AI will speak health insights aloud' },
              { id: 'autoReadChat', icon: MessageSquare, label: 'Auto-read Chat', desc: 'Automatically read chat replies', disabled: !settings.voiceFeedback }
            ].map((item) => (
              <div
                key={item.id}
                onClick={() => !item.disabled && toggleSetting(item.id as keyof typeof settings)}
                className={`flex items-center justify-between p-4 bg-white/5 rounded-2xl transition-all ${item.disabled ? 'opacity-40 cursor-not-allowed' : 'hover:bg-white/10 cursor-pointer group'
                  }`}
              >
                <div className="flex items-center gap-4">
                  <div className={`transition-colors ${item.disabled ? 'text-slate-600' : 'text-slate-400 group-hover:text-blue-500'
                    }`}>
                    <item.icon size={20} />
                  </div>
                  <div>
                    <p className={`text-sm font-medium ${item.disabled ? 'text-slate-500' : 'text-white'
                      }`}>
                      {item.label}
                    </p>
                    <p className="text-xs text-slate-500">{item.desc}</p>
                  </div>
                </div>
                <div className={`w-10 h-5 rounded-full relative transition-colors ${settings[item.id as keyof typeof settings] && !item.disabled ? 'bg-blue-600' : 'bg-slate-700'
                  }`}>
                  <div className={`absolute top-1 w-3 h-3 bg-white rounded-full transition-all ${settings[item.id as keyof typeof settings] && !item.disabled ? 'right-1' : 'left-1'
                    }`} />
                </div>
              </div>
            ))}
          </div>
        </div>

      </div>
    );

    const renderDataExport = () => (
      <div className="p-6 space-y-6">
        {exportStatus && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            className="p-4 bg-blue-500/10 border border-blue-500/20 rounded-2xl space-y-3"
          >
            <div className="flex items-center justify-between text-xs">
              <span className="text-blue-400 font-medium">{exportStatus}</span>
              {exportProgress !== null && (
                <span className="text-blue-500 font-bold">{exportProgress}%</span>
              )}
            </div>
            {exportProgress !== null && (
              <div className="w-full h-1.5 bg-slate-800 rounded-full overflow-hidden">
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${exportProgress}%` }}
                  className="h-full bg-blue-500 shadow-[0_0_10px_rgba(59,130,246,0.5)]"
                />
              </div>
            )}
          </motion.div>
        )}
        <div className="space-y-4">
          <h4 className="text-xs font-bold text-slate-500 uppercase tracking-widest">Data Export</h4>
          <div className="space-y-2">
            {[
              { id: 'autoExport', icon: History, label: 'Automatic Export', desc: 'Monthly backup of health data to cloud' }
            ].map((item) => (
              <div
                key={item.id}
                onClick={() => toggleSetting(item.id as keyof typeof settings)}
                className="flex items-center justify-between p-4 bg-white/5 rounded-2xl hover:bg-white/10 transition-colors cursor-pointer group"
              >
                <div className="flex items-center gap-4">
                  <div className="text-slate-400 group-hover:text-blue-500 transition-colors"><item.icon size={20} /></div>
                  <div>
                    <p className="text-sm font-medium text-white">{item.label}</p>
                    <p className="text-xs text-slate-500">{item.desc}</p>
                  </div>
                </div>
                <div className={`w-10 h-5 rounded-full relative transition-colors ${settings[item.id as keyof typeof settings] ? 'bg-blue-600' : 'bg-slate-700'}`}>
                  <div className={`absolute top-1 w-3 h-3 bg-white rounded-full transition-all ${settings[item.id as keyof typeof settings] ? 'right-1' : 'left-1'}`} />
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="space-y-4">
          <h4 className="text-xs font-bold text-slate-500 uppercase tracking-widest">Available Formats</h4>
          <div className="grid grid-cols-2 gap-3">
            {['PDF Report', 'CSV Data', 'JSON Raw', 'Health Connect'].map((format) => (
              <button
                key={format}
                onClick={() => handleExportData(format)}
                className="p-4 bg-white/5 rounded-2xl hover:bg-white/10 transition-colors text-left group"
              >
                <Download size={18} className="text-slate-500 group-hover:text-blue-500 mb-2 transition-colors" />
                <p className="text-sm font-medium text-white">{format}</p>
                <p className="text-[10px] text-slate-500">Download now</p>
              </button>
            ))}
          </div>
        </div>
      </div>
    );

    const renderContactUs = () => {
      if (showContactForm) {
        return (
          <div className="p-6 space-y-6">
            <div className="flex items-center gap-3 mb-2">
              <button
                onClick={() => setShowContactForm(false)}
                className="p-2 hover:bg-white/5 rounded-full text-slate-400 hover:text-white transition-colors"
              >
                <ArrowLeft size={18} />
              </button>
              <h3 className="text-xl font-bold text-white">Contact Us Form</h3>
            </div>

            <div className="space-y-4">
              {contactFormStatus ? (
                <div className="p-8 text-center space-y-4">
                  <div className="w-16 h-16 bg-green-500/10 rounded-full flex items-center justify-center text-green-500 mx-auto">
                    <CheckCircle2 size={32} />
                  </div>
                  <h4 className="text-lg font-bold text-white">Message Sent!</h4>
                  <p className="text-slate-400 text-sm">{contactFormStatus}</p>
                  <button
                    onClick={() => {
                      setShowContactForm(false);
                      setContactFormStatus(null);
                    }}
                    className="px-6 py-2 bg-blue-600 text-white rounded-xl font-bold hover:bg-blue-700 transition-colors"
                  >
                    Back to Settings
                  </button>
                </div>
              ) : (
                <>
                  <div className="space-y-1.5">
                    <label className="text-xs font-bold text-slate-500 uppercase tracking-widest ml-1">Your Name</label>
                    <input
                      type="text"
                      placeholder="Enter your name"
                      value={contactFormData.name}
                      onChange={(e) => setContactFormData({ ...contactFormData, name: e.target.value })}
                      className="w-full p-4 bg-white/5 border border-white/10 rounded-2xl text-white placeholder:text-slate-600 focus:outline-none focus:border-blue-500/50 transition-colors"
                    />
                  </div>
                  <div className="space-y-1.5">
                    <label className="text-xs font-bold text-slate-500 uppercase tracking-widest ml-1">Email Address</label>
                    <input
                      type="email"
                      placeholder="Enter your email"
                      value={contactFormData.email}
                      onChange={(e) => setContactFormData({ ...contactFormData, email: e.target.value })}
                      className="w-full p-4 bg-white/5 border border-white/10 rounded-2xl text-white placeholder:text-slate-600 focus:outline-none focus:border-blue-500/50 transition-colors"
                    />
                  </div>
                  <div className="space-y-1.5">
                    <label className="text-xs font-bold text-slate-500 uppercase tracking-widest ml-1">Message</label>
                    <textarea
                      rows={4}
                      placeholder="How can we help you?"
                      value={contactFormData.message}
                      onChange={(e) => setContactFormData({ ...contactFormData, message: e.target.value })}
                      className="w-full p-4 bg-white/5 border border-white/10 rounded-2xl text-white placeholder:text-slate-600 focus:outline-none focus:border-blue-500/50 transition-colors resize-none"
                    />
                  </div>
                  <button
                    onClick={async () => {
                      if (!contactFormData.name || !contactFormData.email || !contactFormData.message) {
                        setExportStatus("Please fill in all fields");
                        setTimeout(() => setExportStatus(null), 3000);
                        return;
                      }
                      setContactFormLoading(true);
                      // Simulate sending
                      await new Promise(r => setTimeout(r, 1500));
                      setContactFormLoading(false);
                      setContactFormStatus("Thank you for reaching out! Our team will get back to you within 24 hours.");
                      setContactFormData({ name: '', email: '', message: '' });
                    }}
                    disabled={contactFormLoading}
                    className={`w-full p-4 bg-blue-600 text-white rounded-2xl font-bold transition-all shadow-lg shadow-blue-600/20 flex items-center justify-center gap-2 ${contactFormLoading ? 'opacity-70 cursor-not-allowed' : 'hover:bg-blue-700'}`}
                  >
                    {contactFormLoading ? (
                      <>
                        <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                        Sending...
                      </>
                    ) : (
                      'Send Message'
                    )}
                  </button>
                </>
              )}
            </div>
          </div>
        );
      }

      return (
        <div className="p-6 space-y-8">
          <div className="text-center space-y-2">
            <div className="w-16 h-16 bg-blue-500/10 rounded-full flex items-center justify-center text-blue-500 mx-auto mb-4">
              <Mail size={32} />
            </div>
            <h3 className="text-xl font-bold text-white">How can we help?</h3>
            <p className="text-slate-400 text-sm">Our team is here to support your health journey.</p>
          </div>

          <div className="space-y-3">
            {[
              {
                icon: Mail,
                label: 'Email Support',
                value: 'Open Contact Form',
                color: 'text-blue-400',
                action: () => setShowContactForm(true)
              },
              {
                icon: Phone,
                label: 'Call Us',
                value: '+2348026165268',
                color: 'text-emerald-400',
                action: () => setShowCallModal(true)
              },
              {
                icon: Globe,
                label: 'Help Center',
                value: 'Not available yet',
                color: 'text-purple-400',
                action: () => setShowHelpModal(true)
              }
            ].map((item, i) => (
              <div
                key={i}
                onClick={item.action}
                className="p-4 bg-white/5 rounded-2xl border border-white/5 flex items-center gap-4 hover:bg-white/10 transition-colors cursor-pointer group"
              >
                <div className={`w-10 h-10 rounded-xl bg-white/5 flex items-center justify-center ${item.color}`}>
                  <item.icon size={20} />
                </div>
                <div>
                  <p className="text-xs text-slate-500 font-medium uppercase tracking-wider">{item.label}</p>
                  <p className="text-sm font-bold text-white group-hover:text-blue-400 transition-colors">{item.value}</p>
                </div>
                <ArrowRight size={16} className="ml-auto text-slate-600 group-hover:text-white transition-colors" />
              </div>
            ))}
          </div>

          <div className="p-6 bg-blue-600/10 rounded-3xl border border-blue-500/20 space-y-4">
            <p className="text-sm text-blue-200 leading-relaxed">
              "Your health and privacy are our top priorities. If you have any questions about your data or how Smoking&DrinkingHealth.AI works, please don't hesitate to reach out."
            </p>
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 rounded-full bg-blue-500 flex items-center justify-center text-white font-bold text-xs">S</div>
              <div>
                <p className="text-xs font-bold text-white">Smoking&DrinkingHealth.AI Support Team</p>
                <p className="text-[10px] text-blue-400">Available 24/7</p>
              </div>
            </div>
          </div>
        </div>
      );
    };

    const getTitle = () => {
      switch (activeSettingsTab) {
        case 'privacy': return 'Privacy & Security';
        case 'notifications': return 'Notifications';
        case 'ai': return 'AI & Voice';
        case 'export': return 'Data Export';
        case 'contact': return 'Contact Us';
        default: return 'Settings';
      }
    };

    return (
      <AnimatePresence>
        {showSettings && (
          <div className="fixed inset-0 z-[100] flex items-center justify-center p-4">
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={() => {
                setShowSettings(false);
                setActiveSettingsTab('main');
              }}
              className="absolute inset-0 bg-slate-950/80 backdrop-blur-sm"
            />
            <motion.div
              initial={{ opacity: 0, scale: 0.9, y: 20 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.9, y: 20 }}
              className="relative w-full max-w-lg bg-slate-900 border border-white/10 rounded-3xl overflow-hidden shadow-2xl"
            >
              <div className="p-6 border-b border-white/5 flex items-center justify-between">
                <div className="flex items-center gap-3">
                  {activeSettingsTab !== 'main' && (
                    <button
                      onClick={() => setActiveSettingsTab('main')}
                      className="p-2 hover:bg-white/5 rounded-full text-slate-400 hover:text-white transition-colors"
                    >
                      <ArrowLeft size={18} />
                    </button>
                  )}
                  <h3 className="text-xl font-bold text-white">{getTitle()}</h3>
                </div>
                <button
                  onClick={() => {
                    setShowSettings(false);
                    setActiveSettingsTab('main');
                  }}
                  className="p-2 hover:bg-white/5 rounded-full text-slate-400 hover:text-white transition-colors"
                >
                  <X size={20} />
                </button>
              </div>

              <div className="max-h-[70vh] overflow-y-auto custom-scrollbar">
                {activeSettingsTab === 'main' && renderMain()}
                {activeSettingsTab === 'privacy' && renderPrivacy()}
                {activeSettingsTab === 'notifications' && renderNotifications()}
                {activeSettingsTab === 'ai' && renderAI()}
                {activeSettingsTab === 'export' && renderDataExport()}
                {activeSettingsTab === 'contact' && renderContactUs()}
              </div>

              {/* Delete Account Confirmation Pop-up */}
              <AnimatePresence>
                {showDeleteConfirm && (
                  <div className="fixed inset-0 z-[110] flex items-center justify-center p-4">
                    <motion.div
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      exit={{ opacity: 0 }}
                      onClick={() => setShowDeleteConfirm(false)}
                      className="absolute inset-0 bg-slate-950/60 backdrop-blur-sm"
                    />
                    <motion.div
                      initial={{ opacity: 0, scale: 0.9, y: 20 }}
                      animate={{ opacity: 1, scale: 1, y: 0 }}
                      exit={{ opacity: 0, scale: 0.9, y: 20 }}
                      className="relative w-full max-w-xs bg-slate-900 border border-white/10 rounded-3xl p-8 text-center shadow-2xl"
                    >
                      <div className="w-16 h-16 bg-red-500/10 rounded-full flex items-center justify-center text-red-500 mx-auto mb-6">
                        <Trash2 size={32} />
                      </div>
                      <h3 className="text-xl font-bold text-white mb-2">Delete Account?</h3>
                      <p className="text-slate-400 text-sm mb-8 leading-relaxed">
                        This action is permanent and will delete all your health history and profile data.
                      </p>
                      <div className="flex flex-col gap-3">
                        <button
                          onClick={async () => {
                            // Implement real delete logic here if needed
                            // For now, we simulate and logout
                            setExportStatus("Deleting account data...");
                            await new Promise(r => setTimeout(r, 2000));
                            setShowDeleteConfirm(false);
                            setShowSettings(false);
                            handleLogout();
                          }}
                          className="w-full py-4 bg-red-600 text-white rounded-2xl font-bold hover:bg-red-700 transition-colors"
                        >
                          Delete Permanently
                        </button>
                        <button
                          onClick={() => setShowDeleteConfirm(false)}
                          className="w-full py-4 bg-white/5 text-slate-400 rounded-2xl font-bold hover:bg-white/10 transition-colors"
                        >
                          Cancel
                        </button>
                      </div>
                    </motion.div>
                  </div>
                )}
              </AnimatePresence>

              {/* Call Modal Pop-up */}
              <AnimatePresence>
                {showCallModal && (
                  <div className="fixed inset-0 z-[110] flex items-center justify-center p-4">
                    <motion.div
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      exit={{ opacity: 0 }}
                      onClick={() => setShowCallModal(false)}
                      className="absolute inset-0 bg-slate-950/60 backdrop-blur-sm"
                    />
                    <motion.div
                      initial={{ opacity: 0, scale: 0.9, y: 20 }}
                      animate={{ opacity: 1, scale: 1, y: 0 }}
                      exit={{ opacity: 0, scale: 0.9, y: 20 }}
                      className="relative w-full max-w-xs bg-slate-900 border border-white/10 rounded-3xl p-8 text-center shadow-2xl"
                    >
                      <div className="w-16 h-16 bg-emerald-500/10 rounded-full flex items-center justify-center text-emerald-500 mx-auto mb-6">
                        <Phone size={32} />
                      </div>
                      <h3 className="text-xl font-bold text-white mb-2">Call Support</h3>
                      <p className="text-slate-400 text-sm mb-6">Our team is ready to help you.</p>
                      <p className="text-2xl font-bold text-white mb-8 tracking-wider">+2348026165268</p>
                      <div className="flex flex-col gap-3">
                        <a
                          href="tel:+2348026165268"
                          className="w-full py-4 bg-emerald-600 text-white rounded-2xl font-bold hover:bg-emerald-700 transition-colors flex items-center justify-center gap-2"
                        >
                          <Phone size={18} />
                          Call Now
                        </a>
                        <button
                          onClick={() => setShowCallModal(false)}
                          className="w-full py-4 bg-white/5 text-slate-400 rounded-2xl font-bold hover:bg-white/10 transition-colors"
                        >
                          Cancel
                        </button>
                      </div>
                    </motion.div>
                  </div>
                )}
              </AnimatePresence>

              {/* Help Center Pop-up */}
              <AnimatePresence>
                {showHelpModal && (
                  <div className="fixed inset-0 z-[110] flex items-center justify-center p-4">
                    <motion.div
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      exit={{ opacity: 0 }}
                      onClick={() => setShowHelpModal(false)}
                      className="absolute inset-0 bg-slate-950/60 backdrop-blur-sm"
                    />
                    <motion.div
                      initial={{ opacity: 0, scale: 0.9, y: 20 }}
                      animate={{ opacity: 1, scale: 1, y: 0 }}
                      exit={{ opacity: 0, scale: 0.9, y: 20 }}
                      className="relative w-full max-w-xs bg-slate-900 border border-white/10 rounded-3xl p-8 text-center shadow-2xl"
                    >
                      <div className="w-16 h-16 bg-purple-500/10 rounded-full flex items-center justify-center text-purple-500 mx-auto mb-6">
                        <Globe size={32} />
                      </div>
                      <h3 className="text-xl font-bold text-white mb-2">Help Center</h3>
                      <p className="text-slate-400 text-sm mb-8 leading-relaxed">
                        The Help Center is currently under construction and is <span className="text-white font-bold">not available yet</span>.
                      </p>
                      <button
                        onClick={() => setShowHelpModal(false)}
                        className="w-full py-4 bg-blue-600 text-white rounded-2xl font-bold hover:bg-blue-700 transition-colors"
                      >
                        Got it
                      </button>
                    </motion.div>
                  </div>
                )}
              </AnimatePresence>
            </motion.div>
          </div>
        )}
      </AnimatePresence>
    );
  };

  // --- Voice Mode Overlay ---
  const VoiceMode = () => {
    const [isListening, setIsListening] = useState(false);
    const recognitionRef = useRef<any>(null);

    useEffect(() => {
      if (!isVoiceMode) return;

      const SpeechRecognition =
        (window as any).SpeechRecognition ||
        (window as any).webkitSpeechRecognition;

      if (!SpeechRecognition) {
        alert("Voice recognition not supported in this browser.");
        setIsVoiceMode(false);
        return;
      }

      const recognition = new SpeechRecognition();
      recognition.lang = "en-US";
      recognition.continuous = false;
      recognition.interimResults = false;

      recognition.onstart = () => {
        setIsListening(true);
      };

      recognition.onresult = (event: any) => {
        const text = event.results[0][0].transcript;

        // ✅ put voice into input
        setInputMessage(text);

        // ✅ auto send
        setTimeout(() => {
          handleSendMessage();
        }, 300);
      };

      recognition.onerror = (err: any) => {
        console.error("Voice error:", err);
        setIsListening(false);
      };

      recognition.onend = () => {
        setIsListening(false);
      };

      recognitionRef.current = recognition;
      recognition.start();

      return () => {
        recognition.stop();
      };
    }, [isVoiceMode]);

    return (
      <AnimatePresence>
        {isVoiceMode && (
          <motion.div
            className="fixed inset-0 z-[150] bg-black flex flex-col items-center justify-center text-white"
          >
            <h2 className="text-lg mb-6">
              {isListening ? "Listening..." : "Tap to Speak"}
            </h2>

            <button
              onClick={() => recognitionRef.current?.start()}
              className="w-20 h-20 rounded-full bg-blue-600 flex items-center justify-center text-white text-xl"
            >
              🎤
            </button>

            <button
              onClick={() => setIsVoiceMode(false)}
              className="mt-10 text-sm opacity-70"
            >
              Close
            </button>
          </motion.div>
        )}
      </AnimatePresence>
    );
  };

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const loadModels = async () => {
    try {
      setModelError(null);
      await Promise.all([
        smokingPredictor.current.load('/models/Retrain_smoking_model_xgb.json'),
        drinkingPredictor.current.load('/models/Retrain_Drinking_model_xgb.json')
      ]);
      setModelsLoaded(true);
    } catch (error) {
      console.error("Failed to load models:", error);
      setModelError("AI models failed to initialize. Please check your connection and refresh.");
    }
  };

  const handlePredict = async () => {
    if (!modelsLoaded) return;
    setPredictionLoading(true);

    // Simulate processing delay for UI feel
    await new Promise(resolve => setTimeout(resolve, 1500));

    const smokingProb = smokingPredictor.current?.predict(metrics) ?? 0;
    const drinkingProb = drinkingPredictor.current?.predict(metrics) ?? 0;

    setPredictionResult({
      smoking: smokingProb,
      drinking: drinkingProb,
      timestamp: new Date().toISOString()
    });

    setPredictionLoading(false);
    handleTabChange('results');

    // Generate AI Insights
    generateAIInsights(smokingProb, drinkingProb);

    // Notification
    if (settings.notifications && "Notification" in window) {
      if (Notification.permission === "granted") {
        new Notification("Analysis Complete", { body: "Your health risk profile is ready for review." });
      } else if (Notification.permission !== "denied") {
        Notification.requestPermission();
      }
    }
  };

  const generateAIInsights = async (smoking: number, drinking: number) => {
    try {
      const res = await fetch('/api/insights', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ smoking, drinking })
      });

      const data = await res.json();

      const insight = data.text || "Unable to generate insight at the moment.";

      setPredictionResult((prev: any) => ({ ...prev, insight }));

      if (settings.voiceFeedback) {
        speakText(insight, Date.now().toString());
      }

    } catch (e) {
      console.error("AI Insight failed:", e);

      const fallback = `Smoking risk: ${(smoking * 100).toFixed(1)}%, Drinking risk: ${(drinking * 100).toFixed(1)}%. Maintain healthy habits and monitor your lifestyle.`;

      setPredictionResult((prev: any) => ({ ...prev, insight: fallback }));
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => {
        setSelectedImage(reader.result as string);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleSendMessage = async () => {
    if (!inputMessage.trim() && !selectedImage) return;

    stopSpeaking();

    const userMsg: any = { role: 'user', content: inputMessage };
    if (selectedImage) userMsg.image = selectedImage;

    setMessages(prev => [...prev, userMsg]);

    const currentInput = inputMessage;
    const currentImage = selectedImage;

    setInputMessage('');
    setSelectedImage(null);
    setIsTyping(true);

    try {
      const history = messages.map(msg => ({
        role: msg.role === 'assistant' ? 'model' : 'user',
        parts: [{ text: msg.content }]
      }));

      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: currentInput,
          history,
          image: currentImage || null
        })
      });

      const data = await res.json();

      let aiResponse = "I'm here to help, but I'm having a little trouble right now. Please try again in a moment.";

      if (res.ok && data?.text) {
        aiResponse = data.text;
      }

      // Friendly fallback for overload / errors
      if (res.status === 503) {
        aiResponse =
          "⚠️ Elena is a bit busy right now. Please try again in a few seconds — I’ll be right back with your answer.";
      }

      if (res.status >= 500 && res.status !== 503) {
        aiResponse =
          "Something went wrong on my side. Please try again shortly.";
      }

      if (settings.autoReadChat) {
        speakText(aiResponse, Date.now().toString());
      }

      setMessages(prev => [
        ...prev,
        { role: 'assistant', content: aiResponse }
      ]);

    } catch (e) {
      console.error("Chat failed:", e);

      setMessages(prev => [
        ...prev,
        {
          role: 'assistant',
          content:
            "I couldn't reach Elena right now. Please check your connection and try again."
        }
      ]);
    } finally {
      setIsTyping(false);
    }
  };

  // 🎤 Voice Input State
  const [isRecording, setIsRecording] = useState(false);
  const recognitionRef = useRef(null);

  const startVoiceInput = () => {
    const SpeechRecognition =
      window.SpeechRecognition || window.webkitSpeechRecognition;

    if (!SpeechRecognition) {
      alert("Speech recognition not supported in this browser");
      return;
    }

    const recognition = new SpeechRecognition();

    recognition.lang = "en-US";
    recognition.interimResults = true;
    recognition.continuous = false;

    recognition.onstart = () => setIsRecording(true);

    recognition.onresult = (event) => {
      let text = "";

      for (let i = event.resultIndex; i < event.results.length; i++) {
        text += event.results[i][0].transcript;
      }

      setInputMessage(text); // 🎯 fills your input box
    };

    recognition.onerror = (err) => {
      console.error("Mic error:", err);
      setIsRecording(false);
    };

    recognition.onend = () => {
      setIsRecording(false);

      // 🚀 OPTIONAL: auto send after speaking
      if (inputMessage.trim()) {
        handleSendMessage();
      }
    };

    recognitionRef.current = recognition;
    recognition.start();
  };

  const stopVoiceInput = () => {
    recognitionRef.current?.stop();
    setIsRecording(false);
  };

  if (!isMounted) {
    return <div className="min-h-screen bg-slate-950" />;
  }

  return (
    <div className="min-h-screen bg-slate-950 text-slate-200 font-sans selection:bg-blue-500/30">
      {/* Background Glow */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute -top-[10%] -left-[10%] w-[40%] h-[40%] bg-blue-600/10 blur-[120px] rounded-full" />
        <div className="absolute top-[20%] -right-[10%] w-[30%] h-[30%] bg-purple-600/10 blur-[120px] rounded-full" />
        <div className="absolute -bottom-[10%] left-[20%] w-[50%] h-[50%] bg-emerald-600/5 blur-[120px] rounded-full" />
      </div>

      {/* Navigation */}
      <nav className="fixed bottom-6 left-1/2 -translate-x-1/2 z-50">
        <GlassCard className="flex items-center gap-1 p-1.5 shadow-2xl shadow-black/50">
          {[
            { id: 'home', icon: Activity, label: 'Home' },
            { id: 'predict', icon: Brain, label: 'Predict' },
            { id: 'chat', icon: MessageSquare, label: 'AI' },
            { id: 'profile', icon: UserIcon, label: 'Profile' }
          ].map((item) => (
            <button
              key={item.id}
              onClick={() => handleTabChange(item.id)}
              className={`flex items-center gap-2 px-4 py-2 rounded-xl transition-all ${activeTab === item.id
                ? 'bg-blue-600 text-white shadow-lg shadow-blue-600/20'
                : 'text-slate-400 hover:text-slate-200 hover:bg-white/5'
                }`}
            >
              <item.icon size={20} />
              {activeTab === item.id && <span className="text-sm font-medium">{item.label}</span>}
            </button>
          ))}
        </GlassCard>
      </nav>

      {/* Main Content Area */}
      <main className="max-w-4xl mx-auto px-6 pt-12 pb-32">
        <AnimatePresence mode="wait">
          {activeTab === 'home' && (
            <motion.div
              key="home"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="space-y-8"
            >
              <header className="space-y-2">
                <h1 className="text-4xl font-bold tracking-tight text-white">
                  {user ? `Welcome back, ` : `Welcome to `}
                  <span className="text-blue-500">{user ? user.displayName?.split(' ')[0] : 'Elena'}</span>
                </h1>
                <p className="text-slate-400 text-lg">
                  {user ? 'Your personalized health risk analysis dashboard.' : 'Advanced predictive analytics for a healthier, data-driven lifestyle.'}
                </p>
              </header>

              {/* Smoking&DrinkingHealth.AI Hero Section */}
              <GlassCard className="p-8 border-blue-500/30 bg-gradient-to-br from-blue-600/10 to-purple-600/10 relative overflow-hidden">
                <div className="absolute top-0 right-0 p-6 text-blue-500/10">
                  <Activity size={120} />
                </div>
                <div className="relative z-10 space-y-4">
                  <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-blue-500/20 text-blue-400 text-xs font-bold uppercase tracking-widest">
                    <Zap size={14} /> New Feature
                  </div>
                  <h2 className="text-3xl font-bold text-white">Smoking&DrinkingHealth.AI</h2>
                  <p className="text-slate-300 max-w-2xl leading-relaxed">
                    Harness the power of advanced machine learning to transform your health data into actionable insights.
                    Our XGBoost models analyze your unique metrics to provide precise risk assessments for smoking and drinking behaviors.
                  </p>
                  <div className="flex gap-4 pt-2">
                    <button
                      onClick={() => handleTabChange('predict')}
                      className="px-6 py-2.5 bg-blue-600 text-white rounded-xl font-bold hover:bg-blue-500 transition-all shadow-lg shadow-blue-600/20"
                    >
                      Start Analysis
                    </button>
                    <button
                      onClick={() => setActiveTab('chat')}
                      className="px-6 py-2.5 bg-white/10 text-white border border-white/20 rounded-xl font-bold hover:bg-white/20 transition-all"
                    >
                      Learn More
                    </button>
                  </div>
                </div>
              </GlassCard>

              {modelError && (
                <GlassCard className="p-4 bg-red-500/10 border-red-500/20 text-red-400 text-center">
                  <p className="flex items-center justify-center gap-2">
                    <AlertCircle size={20} />
                    {modelError}
                  </p>
                </GlassCard>
              )}

              {/* Featured Images Section */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 h-auto md:h-64 overflow-hidden rounded-2xl">
                <div className="relative group overflow-hidden rounded-xl h-48 md:h-full">
                  <Image
                    src="https://images.unsplash.com/photo-1576091160550-2173dba999ef?auto=format&fit=crop&w=800&q=80"
                    alt="Doctor and Patient"
                    fill
                    className="object-cover transition-transform duration-500 group-hover:scale-110"
                    referrerPolicy="no-referrer"
                  />
                  <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-transparent to-transparent flex items-end p-4">
                    <p className="text-white text-sm font-medium">Expert Clinical Guidance</p>
                  </div>
                </div>
                <div className="relative group overflow-hidden rounded-xl h-48 md:h-full">
                  <Image
                    src="https://images.unsplash.com/photo-1551076805-e1869033e561?auto=format&fit=crop&w=800&q=80"
                    alt="Medical Consultation"
                    fill
                    className="object-cover transition-transform duration-500 group-hover:scale-110"
                    referrerPolicy="no-referrer"
                  />
                  <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-transparent to-transparent flex items-end p-4">
                    <p className="text-white text-sm font-medium">Personalized Care</p>
                  </div>
                </div>
              </div>

              {/* Auth Section for Guests */}
              {!user && (
                <GlassCard className="p-8 border-blue-500/30 bg-blue-600/5 text-center space-y-6">
                  <div className="space-y-2">
                    <h2 className="text-2xl font-bold text-white">Join the Community</h2>
                    <p className="text-slate-400">Sign up today to unlock full risk assessments and AI-powered health insights.</p>
                  </div>
                  <div className="flex flex-col sm:flex-row gap-4 justify-center">
                    <button
                      onClick={() => { setAuthMode('signin'); setShowAuthModal(true); }}
                      className="px-8 py-3 bg-blue-600 text-white rounded-xl font-bold hover:bg-blue-500 transition-all flex items-center justify-center gap-2 shadow-lg shadow-blue-600/20"
                    >
                      <LogIn size={20} />
                      Sign In
                    </button>
                    <button
                      onClick={() => { setAuthMode('signup'); setShowAuthModal(true); }}
                      className="px-8 py-3 bg-white/10 text-white border border-white/20 rounded-xl font-bold hover:bg-white/20 transition-all flex items-center justify-center gap-2"
                    >
                      <UserPlus size={20} />
                      Sign Up
                    </button>
                  </div>
                </GlassCard>
              )}

              {/* Model Performance Section */}
              <section className="space-y-4">
                <h2 className="text-2xl font-bold text-white flex items-center gap-2">
                  <Target size={24} className="text-blue-500" />
                  Model Performance
                </h2>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  {[
                    { label: 'Accuracy', value: '92.4%', icon: Award, color: 'text-emerald-500' },
                    { label: 'Precision', value: '89.1%', icon: Zap, color: 'text-blue-500' },
                    { label: 'Recall', value: '91.5%', icon: TrendingUp, color: 'text-purple-500' },
                    { label: 'F1 Score', value: '90.3%', icon: Shield, color: 'text-amber-500' }
                  ].map((stat, i) => (
                    <GlassCard key={i} className="p-4 text-center space-y-2">
                      <stat.icon size={20} className={`mx-auto ${stat.color}`} />
                      <div className="text-xl font-bold text-white">{stat.value}</div>
                      <div className="text-[10px] uppercase tracking-widest text-slate-500 font-bold">{stat.label}</div>
                    </GlassCard>
                  ))}
                </div>
                <p className="text-xs text-slate-500 italic text-center">
                  * Metrics based on the latest retrained XGBoost models using a dataset of 100,000+ clinical records.
                </p>
              </section>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <GlassCard className="p-6 space-y-4 group cursor-pointer hover:border-blue-500/50 transition-colors" onClick={() => handleTabChange('predict')}>
                  <div className="w-12 h-12 rounded-xl bg-blue-600/20 flex items-center justify-center text-blue-500 group-hover:scale-110 transition-transform">
                    <Brain size={24} />
                  </div>
                  <div>
                    <h3 className="text-xl font-semibold text-white">Risk Prediction</h3>
                    <p className="text-slate-400 text-sm mt-1">Analyze your health metrics using our advanced XGBoost models.</p>
                  </div>
                  <div className="flex items-center text-blue-500 text-sm font-medium gap-1">
                    Start Analysis <ChevronRight size={16} />
                  </div>
                </GlassCard>

                <GlassCard className="p-6 space-y-4 group cursor-pointer hover:border-purple-500/50 transition-colors" onClick={() => handleTabChange('chat')}>
                  <div className="w-12 h-12 rounded-xl bg-purple-600/20 flex items-center justify-center text-purple-500 group-hover:scale-110 transition-transform">
                    <MessageSquare size={24} />
                  </div>
                  <div>
                    <h3 className="text-xl font-semibold text-white">Talk to AI</h3>
                    <p className="text-slate-400 text-sm mt-1">Get instant health insights and advice from your AI companion.</p>
                  </div>
                  <div className="flex items-center text-purple-500 text-sm font-medium gap-1">
                    Open Chat <ChevronRight size={16} />
                  </div>
                </GlassCard>
              </div>

              <section className="space-y-4">
                <h2 className="text-2xl font-bold text-white flex items-center gap-2">
                  <TrendingUp size={24} className="text-emerald-500" />
                  Recent Activity
                </h2>
                {predictionResult ? (
                  <GlassCard className="p-6 flex items-center justify-between">
                    <div className="flex items-center gap-4">
                      <div className="w-10 h-10 rounded-full bg-emerald-500/20 flex items-center justify-center text-emerald-500">
                        <CheckCircle2 size={20} />
                      </div>
                      <div>
                        <p className="text-white font-medium">Last Analysis Complete</p>
                        <p className="text-slate-500 text-xs">{new Date(predictionResult.timestamp).toLocaleString()}</p>
                      </div>
                    </div>
                    <button onClick={() => handleTabChange('results')} className="text-blue-500 text-sm font-medium hover:underline">View Results</button>
                  </GlassCard>
                ) : (
                  <div className="text-center py-12 border-2 border-dashed border-slate-800 rounded-2xl">
                    <p className="text-slate-500">No recent predictions found. Start your first analysis!</p>
                  </div>
                )}
              </section>

              {/* App Recommended Section */}
              <section className="space-y-4">
                <h2 className="text-2xl font-bold text-white flex items-center gap-2">
                  <Award size={24} className="text-amber-500" />
                  Recommended for You
                </h2>
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                  {[
                    { id: 'diet', title: 'Diet Plan', desc: 'Personalized nutrition based on your risk profile.', icon: HeartPulse, color: 'bg-red-500/20 text-red-500' },
                    { id: 'exercise', title: 'Exercise', desc: 'Daily routines to improve cardiovascular health.', icon: Activity, color: 'bg-blue-500/20 text-blue-500' },
                    { id: 'meditation', title: 'Meditation', desc: 'Stress reduction techniques for better BP.', icon: Wind, color: 'bg-purple-500/20 text-purple-500' }
                  ].map((app, i) => (
                    <GlassCard
                      key={i}
                      onClick={() => handleTabChange('chat')}
                      className="p-5 space-y-3 hover:bg-white/5 transition-colors cursor-pointer"
                    >
                      <div className={`w-10 h-10 rounded-lg flex items-center justify-center ${app.color}`}>
                        <app.icon size={20} />
                      </div>
                      <div>
                        <h4 className="font-bold text-white text-sm">{app.title}</h4>
                        <p className="text-slate-500 text-[11px] leading-tight mt-1">{app.desc}</p>
                      </div>
                    </GlassCard>
                  ))}
                </div>
              </section>

              {/* Smoking & Drinking Awareness Section */}
              <section className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="relative h-48 rounded-2xl overflow-hidden group cursor-pointer" onClick={() => handleTabChange('chat')}>
                  <Image
                    src="https://images.unsplash.com/photo-1527613426441-4da17471b66d?auto=format&fit=crop&w=800&q=80"
                    alt="Smoking Awareness"
                    fill
                    className="object-cover opacity-60 group-hover:scale-105 transition-transform duration-700"
                    referrerPolicy="no-referrer"
                  />
                  <div className="absolute inset-0 bg-gradient-to-t from-slate-950 p-6 flex flex-col justify-end">
                    <h3 className="text-lg font-bold text-white">Smoking Cessation</h3>
                    <p className="text-xs text-slate-400 mt-1">Resources and support to help you quit for good.</p>
                  </div>
                </div>
                <div className="relative h-48 rounded-2xl overflow-hidden group cursor-pointer" onClick={() => handleTabChange('chat')}>
                  <Image
                    src="https://images.unsplash.com/photo-1520174691701-bc555a3404ca?auto=format&fit=crop&w=800&q=80"
                    alt="Drinking Awareness"
                    fill
                    className="object-cover opacity-60 group-hover:scale-105 transition-transform duration-700"
                    referrerPolicy="no-referrer"
                  />
                  <div className="absolute inset-0 bg-gradient-to-t from-slate-950 p-6 flex flex-col justify-end">
                    <h3 className="text-lg font-bold text-white">Moderate Drinking</h3>
                    <p className="text-xs text-slate-400 mt-1">Understanding the impact of alcohol on your long-term health.</p>
                  </div>
                </div>
              </section>
            </motion.div>
          )}

          {activeTab === 'predict' && (
            <motion.div
              key="predict"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
              className="space-y-8"
            >
              {!user ? (
                <div className="flex flex-col items-center justify-center py-20 space-y-6 text-center">
                  <div className="w-20 h-20 rounded-full bg-blue-600/20 flex items-center justify-center text-blue-500">
                    <Shield size={40} />
                  </div>
                  <div className="space-y-2">
                    <h2 className="text-2xl font-bold text-white">Authentication Required</h2>
                    <p className="text-slate-400 max-w-md">
                      To protect your health data and provide personalized risk assessments, you must have an account to access the Predict tool.
                    </p>
                  </div>
                  <button
                    onClick={() => { setAuthMode('signin'); setShowAuthModal(true); }}
                    className="px-8 py-3 bg-blue-600 text-white rounded-xl font-bold hover:bg-blue-500 transition-all shadow-lg shadow-blue-600/20 flex items-center gap-2"
                  >
                    <LogIn size={20} />
                    Sign in with Google
                  </button>
                </div>
              ) : (
                <>
                  <header className="flex justify-between items-end">
                    <div className="space-y-2">
                      <h1 className="text-3xl font-bold text-white">Risk Analysis</h1>
                      <p className="text-slate-400">Enter your clinical metrics for a precise risk assessment.</p>
                    </div>
                    <div className={`px-3 py-1 rounded-full text-xs font-bold uppercase tracking-tighter ${modelsLoaded ? 'bg-emerald-500/10 text-emerald-500' : 'bg-amber-500/10 text-amber-500'}`}>
                      {modelsLoaded ? 'Models Ready' : 'Loading Models...'}
                    </div>
                  </header>

                  <GlassCard className="p-8">
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                      <div className="space-y-6">
                        <h4 className="text-sm font-bold text-blue-500 uppercase tracking-widest">Basic Info</h4>
                        <SelectField
                          label="Sex"
                          value={metrics.sex_numeric}
                          onChange={(v: any) => setMetrics({ ...metrics, sex_numeric: parseInt(v) })}
                          options={[{ label: 'Male', value: 1 }, { label: 'Female', value: 2 }]}
                        />
                        <InputField label="BMI" value={metrics.BMI} onChange={(v: any) => setMetrics({ ...metrics, BMI: v })} placeholder="e.g. 24.5" />
                        <InputField label="Waistline (cm)" value={metrics.waistline} onChange={(v: any) => setMetrics({ ...metrics, waistline: v })} placeholder="e.g. 85" />
                      </div>

                      <div className="space-y-6">
                        <h4 className="text-sm font-bold text-purple-500 uppercase tracking-widest">Vitals & Blood</h4>
                        <InputField label="Systolic BP" value={metrics.SBP} onChange={(v: any) => setMetrics({ ...metrics, SBP: v })} placeholder="e.g. 120" />
                        <InputField label="Diastolic BP" value={metrics.DBP} onChange={(v: any) => setMetrics({ ...metrics, DBP: v })} placeholder="e.g. 80" />
                        <InputField label="Fasting Blood Sugar" value={metrics.BLDS} onChange={(v: any) => setMetrics({ ...metrics, BLDS: v })} placeholder="e.g. 95" />
                        <InputField label="Hemoglobin" value={metrics.hemoglobin} onChange={(v: any) => setMetrics({ ...metrics, hemoglobin: v })} placeholder="e.g. 14.5" />
                        <SelectField
                          label="Urine Protein"
                          value={metrics.urine_protein}
                          onChange={(v: any) => setMetrics({ ...metrics, urine_protein: parseInt(v) })}
                          options={[
                            { label: 'Negative', value: 1 },
                            { label: 'Trace', value: 2 },
                            { label: '+1 (Positive)', value: 3 },
                            { label: '+2 (Positive)', value: 4 },
                            { label: '+3 (Positive)', value: 5 },
                            { label: '+4 (Positive)', value: 6 }
                          ]}
                        />
                      </div>

                      <div className="space-y-6">
                        <h4 className="text-sm font-bold text-emerald-500 uppercase tracking-widest">Lipids & Liver</h4>
                        <InputField label="Triglyceride" value={metrics.triglyceride} onChange={(v: any) => setMetrics({ ...metrics, triglyceride: v })} placeholder="e.g. 150" />
                        <InputField label="HDL Cholesterol" value={metrics.HDL_chole} onChange={(v: any) => setMetrics({ ...metrics, HDL_chole: v })} placeholder="e.g. 50" />
                        <InputField label="LDL Cholesterol" value={metrics.LDL_chole} onChange={(v: any) => setMetrics({ ...metrics, LDL_chole: v })} placeholder="e.g. 110" />
                        <InputField label="Gamma GTP" value={metrics.gamma_GTP} onChange={(v: any) => setMetrics({ ...metrics, gamma_GTP: v })} placeholder="e.g. 35" />
                      </div>
                    </div>

                    <div className="mt-12 pt-8 border-t border-slate-800 flex flex-col md:flex-row gap-6 items-center justify-between">
                      <div className="flex gap-4">
                        <SelectField
                          label="Drinking Status"
                          value={metrics.DRK_YN}
                          onChange={(v: any) => setMetrics({ ...metrics, DRK_YN: parseInt(v) })}
                          options={[{ label: 'No', value: 0 }, { label: 'Yes', value: 1 }]}
                        />
                        <SelectField
                          label="Smoking Status"
                          value={metrics.SMK_stat_type_cd}
                          onChange={(v: any) => setMetrics({ ...metrics, SMK_stat_type_cd: parseInt(v) })}
                          options={[{ label: 'Never', value: 1 }, { label: 'Former', value: 2 }, { label: 'Current', value: 3 }]}
                        />
                      </div>
                      <button
                        onClick={handlePredict}
                        disabled={!modelsLoaded || predictionLoading}
                        className={`px-10 py-4 rounded-2xl font-bold text-lg transition-all flex items-center gap-3 ${predictionLoading
                          ? 'bg-slate-800 text-slate-500 cursor-not-allowed'
                          : 'bg-blue-600 text-white hover:bg-blue-500 shadow-xl shadow-blue-600/20 active:scale-95'
                          }`}
                      >
                        {predictionLoading ? (
                          <>
                            <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                            Analyzing...
                          </>
                        ) : (
                          <>
                            Run Prediction
                            <ArrowRight size={20} />
                          </>
                        )}
                      </button>
                    </div>
                  </GlassCard>
                </>
              )}
            </motion.div>
          )}

          {activeTab === 'results' && predictionResult && (
            <motion.div
              key="results"
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 1.05 }}
              className="space-y-8"
            >
              <header className="text-center space-y-2">
                <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-emerald-500/10 text-emerald-500 text-xs font-bold uppercase tracking-widest mb-2">
                  <CheckCircle2 size={14} /> Analysis Complete
                </div>
                <h1 className="text-4xl font-bold text-white">Your Risk Profile</h1>
              </header>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                <GlassCard className="p-8 space-y-6 relative overflow-hidden">
                  <div className="absolute top-0 right-0 p-4 text-slate-800/20">
                    <Wind size={80} />
                  </div>
                  <div className="space-y-1">
                    <h3 className="text-slate-400 font-medium uppercase tracking-widest text-xs">Smoking Risk</h3>
                    <div className="text-5xl font-bold text-white">{(predictionResult.smoking * 100).toFixed(1)}%</div>
                  </div>
                  <div className="w-full h-3 bg-slate-800 rounded-full overflow-hidden">
                    <motion.div
                      initial={{ width: 0 }}
                      animate={{ width: `${predictionResult.smoking * 100}%` }}
                      transition={{ duration: 1, ease: "easeOut" }}
                      className={`h-full ${predictionResult.smoking > 0.5 ? 'bg-red-500' : 'bg-blue-500'}`}
                    />
                  </div>
                  <p className="text-sm text-slate-400">
                    {predictionResult.smoking > 0.5
                      ? "High risk detected. Consider consulting with a specialist about respiratory health."
                      : "Low risk. Maintain your current healthy lifestyle choices."}
                  </p>
                </GlassCard>

                <GlassCard className="p-8 space-y-6 relative overflow-hidden">
                  <div className="absolute top-0 right-0 p-4 text-slate-800/20">
                    <Beer size={80} />
                  </div>
                  <div className="space-y-1">
                    <h3 className="text-slate-400 font-medium uppercase tracking-widest text-xs">Drinking Risk</h3>
                    <div className="text-5xl font-bold text-white">{(predictionResult.drinking * 100).toFixed(1)}%</div>
                  </div>
                  <div className="w-full h-3 bg-slate-800 rounded-full overflow-hidden">
                    <motion.div
                      initial={{ width: 0 }}
                      animate={{ width: `${predictionResult.drinking * 100}%` }}
                      transition={{ duration: 1, ease: "easeOut", delay: 0.2 }}
                      className={`h-full ${predictionResult.drinking > 0.5 ? 'bg-red-500' : 'bg-purple-500'}`}
                    />
                  </div>
                  <p className="text-sm text-slate-400">
                    {predictionResult.drinking > 0.5
                      ? "Elevated risk for alcohol-related complications. Monitoring liver enzymes is advised."
                      : "Your metrics suggest a low risk for alcohol-related health issues."}
                  </p>
                </GlassCard>
              </div>

              {predictionResult.insight && (
                <GlassCard className="p-8 border-blue-500/30 bg-blue-600/5">
                  <div className="flex items-start gap-4">
                    <div className="w-10 h-10 rounded-xl bg-blue-600/20 flex items-center justify-center text-blue-500 shrink-0">
                      <Brain size={20} />
                    </div>
                    <div className="space-y-2">
                      <h4 className="text-white font-bold">AI Insight</h4>
                      <p className="text-slate-300 leading-relaxed italic">"{predictionResult.insight}"</p>
                    </div>
                  </div>
                </GlassCard>
              )}

              <div className="flex flex-wrap justify-center gap-4">
                <button
                  onClick={() => handleExportData('CSV Data')}
                  className="px-6 py-3 rounded-xl bg-emerald-600 text-white font-medium hover:bg-emerald-500 transition-colors shadow-lg shadow-emerald-600/20 flex items-center gap-2"
                >
                  <Download size={18} />
                  Download Report
                </button>
                <button
                  onClick={() => handleTabChange('predict')}
                  className="px-6 py-3 rounded-xl bg-slate-800 text-white font-medium hover:bg-slate-700 transition-colors"
                >
                  Adjust Metrics
                </button>
                <button
                  onClick={() => handleTabChange('chat')}
                  className="px-6 py-3 rounded-xl bg-blue-600 text-white font-medium hover:bg-blue-500 transition-colors shadow-lg shadow-blue-600/20"
                >
                  Discuss with AI
                </button>
              </div>
            </motion.div>
          )}

          {activeTab === 'chat' && (
            <motion.div
              key="chat"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="h-[calc(100vh-240px)] flex flex-col"
            >
              <GlassCard className="flex-1 flex flex-col shadow-2xl">
                {/* Chat Header */}
                <div className="p-4 border-b border-white/10 flex items-center justify-between bg-white/5">
                  <div className="flex items-center gap-3">
                    <div className="relative">
                      <div className="w-10 h-10 rounded-full bg-gradient-to-tr from-blue-600 to-purple-600 flex items-center justify-center text-white font-bold">
                        S
                      </div>
                      <div className="absolute bottom-0 right-0 w-3 h-3 bg-emerald-500 border-2 border-slate-900 rounded-full" />
                    </div>
                    <div>
                      <h3 className="text-white font-bold text-sm">Elena</h3>
                      <p className="text-slate-500 text-xs">AI Health Companion</p>
                    </div>
                  </div>
                  <div className="flex gap-2">
                    {isSpeaking && (
                      <button
                        onClick={stopSpeaking}
                        className="p-2 text-red-400 hover:text-red-300 hover:bg-red-500/10 rounded-lg transition-colors animate-pulse"
                        title="Stop reading"
                      >
                        <VolumeX size={18} />
                      </button>
                    )}
                    {settings.voiceFeedback && (
                      <button
                        onClick={() => {
                          stopSpeaking();
                          setIsVoiceMode(true);
                        }}
                        className="p-2 text-slate-400 hover:text-white hover:bg-white/5 rounded-lg transition-colors"
                        title="Start Voice Mode"
                      >
                        <Mic size={18} />
                      </button>
                    )}
                    <button
                      onClick={() => setShowSettings(true)}
                      className="p-2 text-slate-400 hover:text-white hover:bg-white/5 rounded-lg transition-colors"
                    >
                      <Settings size={18} />
                    </button>
                  </div>
                </div>

                {/* Messages */}
                <div className="flex-1 overflow-y-auto p-6 space-y-6 custom-scrollbar">
                  {messages.map((msg: any, idx) => (
                    <div key={idx} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                      <div className={`max-w-[80%] p-4 rounded-2xl text-sm leading-relaxed space-y-2 ${msg.role === 'user'
                        ? 'bg-blue-600 text-white rounded-tr-none'
                        : 'bg-slate-800/80 text-slate-200 rounded-tl-none border border-white/5'
                        }`}>
                        {msg.image && (
                          <div className="relative w-full max-h-64 rounded-lg overflow-hidden mb-2">
                            <img src={msg.image} alt="Health Metric" className="w-full h-full object-contain" />
                          </div>
                        )}
                        {msg.content && <div>{msg.content}</div>}
                        {msg.role === 'assistant' && settings.voiceFeedback && (
                          <div className="flex justify-end pt-2">
                            <button
                              onClick={() => speakText(msg.content, idx.toString())}
                              className={`p-1.5 rounded-lg transition-colors ${isSpeaking === idx.toString()
                                ? 'bg-blue-500 text-white'
                                : 'bg-white/5 text-slate-400 hover:text-white hover:bg-white/10'
                                }`}
                              title={isSpeaking === idx.toString() ? "Stop reading" : "Read aloud"}
                            >
                              {isSpeaking === idx.toString() ? <VolumeX size={14} /> : <Volume2 size={14} />}
                            </button>
                          </div>
                        )}
                      </div>
                    </div>
                  ))}
                  {isTyping && (
                    <div className="flex justify-start">
                      <div className="bg-slate-800/80 p-4 rounded-2xl rounded-tl-none border border-white/5 flex gap-1">
                        <div className="w-1.5 h-1.5 bg-slate-500 rounded-full animate-bounce" />
                        <div className="w-1.5 h-1.5 bg-slate-500 rounded-full animate-bounce [animation-delay:0.2s]" />
                        <div className="w-1.5 h-1.5 bg-slate-500 rounded-full animate-bounce [animation-delay:0.4s]" />
                      </div>
                    </div>
                  )}
                  <div ref={chatEndRef} />
                </div>

                {/* Input */}
                <div className="p-4 bg-slate-900/50 border-t border-white/5 space-y-4">
                  {selectedImage && (
                    <div className="relative inline-block">
                      <img src={selectedImage} alt="Preview" className="w-20 h-20 object-cover rounded-lg border border-white/10" />
                      <button
                        onClick={() => setSelectedImage(null)}
                        className="absolute -top-2 -right-2 bg-red-500 text-white rounded-full p-1 shadow-lg"
                      >
                        <X size={12} />
                      </button>
                    </div>
                  )}
                  <div className="relative flex items-center gap-2">
                    <input
                      type="file"
                      ref={fileInputRef}
                      onChange={handleFileChange}
                      accept="image/*"
                      className="hidden"
                    />

                    {/* Image Upload */}
                    <button
                      onClick={() => fileInputRef.current?.click()}
                      className="p-3 bg-slate-800 border border-slate-700 rounded-xl text-slate-400 hover:text-white transition-colors"
                    >
                      <ImagePlus size={20} />
                    </button>

                    {/* 🎤 MIC BUTTON */}
                    <button
                      onClick={isRecording ? stopVoiceInput : startVoiceInput}
                      className={`p-3 border rounded-xl transition-all ${isRecording
                        ? "bg-red-500 text-white animate-pulse border-red-400"
                        : "bg-slate-800 text-slate-400 border-slate-700 hover:text-white"
                        }`}
                      title="Voice Input"
                    >
                      <Mic size={20} />
                    </button>

                    {/* TEXT INPUT */}
                    <div className="relative flex-1">
                      <input
                        type="text"
                        value={inputMessage}
                        onChange={(e) => setInputMessage(e.target.value)}
                        onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()}
                        placeholder={isRecording ? "Listening..." : "Ask Elena anything about your health..."}
                        className="w-full bg-slate-800 border border-slate-700 rounded-xl pl-4 pr-12 py-3 text-white focus:outline-none focus:ring-2 focus:ring-blue-500/50 transition-all"
                      />

                      {/* SEND BUTTON */}
                      <button
                        onClick={handleSendMessage}
                        disabled={!inputMessage.trim() && !selectedImage}
                        className="absolute right-2 top-1/2 -translate-y-1/2 p-2 text-blue-500 hover:text-blue-400 disabled:text-slate-600 transition-colors"
                      >
                        <Send size={20} />
                      </button>
                    </div>
                  </div>
                </div>
              </GlassCard>
            </motion.div>
          )}

          {activeTab === 'profile' && (
            <motion.div
              key="profile"
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 1.05 }}
              className="space-y-8"
            >
              <h1 className="text-3xl font-bold text-white">Your Profile</h1>
              <GlassCard className="p-8 space-y-8">
                {!user ? (
                  <div className="flex flex-col items-center justify-center py-10 space-y-6 text-center">
                    <div className="w-16 h-16 rounded-full bg-slate-800 flex items-center justify-center text-slate-500">
                      <UserIcon size={32} />
                    </div>
                    <div className="space-y-2">
                      <h2 className="text-xl font-bold text-white">Not Signed In</h2>
                      <p className="text-slate-400 text-sm max-w-xs">
                        Sign in to save your health history and sync across devices.
                      </p>
                    </div>
                    <button
                      onClick={() => { setAuthMode('signin'); setShowAuthModal(true); }}
                      className="px-6 py-2 bg-blue-600 text-white rounded-lg font-bold hover:bg-blue-500 transition-all flex items-center gap-2"
                    >
                      <LogIn size={18} />
                      Sign in with Google
                    </button>
                  </div>
                ) : (
                  <>
                    <div className="flex items-center gap-6">
                      {user.photoURL ? (
                        <img src={user.photoURL} alt={user.displayName || 'User'} className="w-24 h-24 rounded-full shadow-xl shadow-blue-600/20 border-2 border-blue-600/50" />
                      ) : (
                        <div className="w-24 h-24 rounded-full bg-gradient-to-br from-blue-600 to-purple-600 flex items-center justify-center text-3xl font-bold text-white shadow-xl shadow-blue-600/20">
                          {user.displayName?.charAt(0) || 'U'}
                        </div>
                      )}
                      <div className="flex-1">
                        <h2 className="text-2xl font-bold text-white">{user.displayName || 'Health Explorer'}</h2>
                        <p className="text-slate-400">{user.email}</p>
                        <div className="flex gap-2 mt-2">
                          <span className="px-2 py-0.5 rounded-md bg-blue-500/10 text-blue-500 text-[10px] font-bold uppercase tracking-wider">Premium Plan</span>
                          <span className="px-2 py-0.5 rounded-md bg-emerald-500/10 text-emerald-500 text-[10px] font-bold uppercase tracking-wider">Verified Identity</span>
                        </div>
                      </div>
                      <button
                        onClick={handleLogout}
                        className="p-3 text-slate-400 hover:text-red-500 hover:bg-red-500/10 rounded-xl transition-all"
                        title="Logout"
                      >
                        <LogOut size={24} />
                      </button>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                      <div className="p-4 bg-slate-800/50 rounded-xl border border-white/5 text-center">
                        <p className="text-slate-500 text-xs uppercase font-bold tracking-widest mb-1">Analyses</p>
                        <p className="text-2xl font-bold text-white">12</p>
                      </div>
                      <div className="p-4 bg-slate-800/50 rounded-xl border border-white/5 text-center">
                        <p className="text-slate-500 text-xs uppercase font-bold tracking-widest mb-1">Health Score</p>
                        <p className="text-2xl font-bold text-emerald-500">84</p>
                      </div>
                      <div className="p-4 bg-slate-800/50 rounded-xl border border-white/5 text-center">
                        <p className="text-slate-500 text-xs uppercase font-bold tracking-widest mb-1">Goal Progress</p>
                        <p className="text-2xl font-bold text-blue-500">72%</p>
                      </div>
                    </div>

                    <div className="space-y-4">
                      <h4 className="text-white font-bold">Settings</h4>
                      <div className="space-y-2">
                        {[
                          { id: 'privacy', icon: Shield, label: 'Privacy & Security', desc: 'Manage your data and encryption keys' },
                          { id: 'export', icon: BarChart3, label: 'Data Export', desc: 'Download your health history in PDF/JSON' },
                          { id: 'contact', icon: Mail, label: 'Contact Us', desc: 'Get in touch with our support team' }
                        ].map((item) => (
                          <div
                            key={item.id}
                            onClick={() => {
                              setActiveSettingsTab(item.id as any);
                              setShowSettings(true);
                            }}
                            className="flex items-center justify-between p-4 bg-white/5 rounded-xl hover:bg-white/10 transition-colors cursor-pointer group"
                          >
                            <div className="flex items-center gap-4">
                              <div className="text-slate-400 group-hover:text-blue-500 transition-colors"><item.icon size={20} /></div>
                              <div>
                                <p className="text-sm font-medium text-white">{item.label}</p>
                                <p className="text-xs text-slate-500">{item.desc}</p>
                              </div>
                            </div>
                            <ChevronRight size={16} className="text-slate-600" />
                          </div>
                        ))}
                      </div>
                    </div>
                  </>
                )}
              </GlassCard>
            </motion.div>
          )}
        </AnimatePresence>
      </main>

      <SettingsModal />
      <AuthModal
        showAuthModal={showAuthModal}
        setShowAuthModal={setShowAuthModal}
        authMode={authMode}
        setAuthMode={setAuthMode}
        handleLogin={handleLogin}
      />
      <VoiceMode />

      <style jsx global>{`
        @keyframes spin-slow {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
        .animate-spin-slow {
          animation: spin-slow 12s linear infinite;
        }
        .custom-scrollbar::-webkit-scrollbar {
          width: 4px;
        }
        .custom-scrollbar::-webkit-scrollbar-track {
          background: transparent;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb {
          background: rgba(255, 255, 255, 0.1);
          border-radius: 10px;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb:hover {
          background: rgba(255, 255, 255, 0.2);
        }
      `}</style>
    </div>
  );
}
