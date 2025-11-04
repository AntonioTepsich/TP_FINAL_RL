"""
🔍 DIAGNÓSTICO 2: Verificar métricas PPO durante entrenamiento
Añade TODOS los indicadores críticos que faltan en tu código actual
"""
import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np

class PPODiagnostic:
    """Versión extendida de PPO con logging completo de métricas críticas"""
    
    def __init__(self, model, lr=3e-4, clip_eps=0.2, ent_coef=0.01, vf_coef=0.5, 
                 max_grad_norm=0.5, device="cuda", verbose=True):
        self.model = model.to(device)
        self.opt = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-5)
        self.clip_eps = clip_eps
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.device = device
        self.verbose = verbose
        
        # Para métricas
        self.update_count = 0
    
    def update(self, batch, epochs=4, minibatch_size=1024):
        obs, acts, logps_old, returns, advs, vals_old = [x.to(self.device) for x in batch]
        
        # Normalizar advantaes
        advs_normalized = (advs - advs.mean()) / (advs.std() + 1e-8)
        
        # Métricas acumuladas
        metrics = {
            'policy_loss': [], 'value_loss': [], 'entropy': [],
            'kl_div': [], 'clip_frac': [], 'ratio_mean': [], 'ratio_std': [],
            'approx_kl': [], 'explained_var': []
        }
        
        n = obs.size(0)
        idxs = torch.arange(n)
        
        for epoch in range(epochs):
            perm = idxs[torch.randperm(n)]
            
            for i in range(0, n, minibatch_size):
                mb = perm[i:i+minibatch_size]
                
                # Forward pass
                logits, v = self.model(obs[mb])
                dist = Categorical(logits=logits)
                logps = dist.log_prob(acts[mb])
                
                # Ratio y KL
                ratio = torch.exp(logps - logps_old[mb])
                kl = logps_old[mb] - logps  # KL aproximado
                
                # Policy loss (PPO clip)
                surr1 = ratio * advs_normalized[mb]
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advs_normalized[mb]
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss (con clipping)
                v_pred = v.squeeze(-1)
                v_clipped = vals_old[mb] + (v_pred - vals_old[mb]).clamp(-self.clip_eps, self.clip_eps)
                vf_losses1 = (v_pred - returns[mb]) ** 2
                vf_losses2 = (v_clipped - returns[mb]) ** 2
                value_loss = 0.5 * torch.max(vf_losses1, vf_losses2).mean()
                
                # Entropy
                entropy = dist.entropy().mean()
                
                # Loss total
                loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy
                
                # Backward
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.opt.step()
                
                # Métricas
                with torch.no_grad():
                    clip_frac = ((ratio - 1.0).abs() > self.clip_eps).float().mean()
                    approx_kl = ((ratio - 1.0) - (ratio.log())).mean()
                    
                    metrics['policy_loss'].append(policy_loss.item())
                    metrics['value_loss'].append(value_loss.item())
                    metrics['entropy'].append(entropy.item())
                    metrics['kl_div'].append(kl.mean().item())
                    metrics['approx_kl'].append(approx_kl.item())
                    metrics['clip_frac'].append(clip_frac.item())
                    metrics['ratio_mean'].append(ratio.mean().item())
                    metrics['ratio_std'].append(ratio.std().item())
        
        # Explained variance
        with torch.no_grad():
            y_pred = vals_old.cpu().numpy()
            y_true = returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = 1 - np.var(y_true - y_pred) / (var_y + 1e-8)
            metrics['explained_var'] = explained_var
        
        # Promediar métricas
        avg_metrics = {
            k: (np.mean(v) if isinstance(v, list) else v) 
            for k, v in metrics.items()
        }
        
        self.update_count += 1
        
        # Log cada N updates
        if self.verbose and self.update_count % 5 == 0:
            self._print_diagnostics(avg_metrics)
        
        return avg_metrics
    
    def _print_diagnostics(self, m):
        """Imprime diagnósticos con interpretación"""
        print(f"\n{'='*70}")
        print(f"🔧 PPO Update #{self.update_count}")
        print(f"{'='*70}")
        
        # Policy
        print(f"📊 POLICY:")
        print(f"   Loss:       {m['policy_loss']:>8.4f}")
        print(f"   Ratio:      {m['ratio_mean']:>8.4f} ± {m['ratio_std']:.4f}", end="")
        if m['ratio_mean'] < 0.8 or m['ratio_mean'] > 1.2:
            print(f"  ⚠️  FUERA DE RANGO!")
        else:
            print(f"  ✅")
        
        print(f"   KL div:     {m['kl_div']:>8.4f}", end="")
        if abs(m['kl_div']) > 0.02:
            print(f"  ⚠️  ALTO (bajá LR o epochs)")
        else:
            print(f"  ✅")
        
        print(f"   Approx KL:  {m['approx_kl']:>8.4f}")
        print(f"   Clip frac:  {m['clip_frac']:>8.4f}", end="")
        if m['clip_frac'] < 0.05:
            print(f"  💡 Muy bajo (podés subir LR)")
        elif m['clip_frac'] > 0.5:
            print(f"  ⚠️  Muy alto (bajá LR o epochs)")
        else:
            print(f"  ✅")
        
        # Value
        print(f"\n📈 VALUE:")
        print(f"   Loss:       {m['value_loss']:>8.4f}")
        print(f"   Expl. Var:  {m['explained_var']:>8.4f}", end="")
        if m['explained_var'] < 0:
            print(f"  ❌ NEGATIVO (crítico aprende mal!)")
        elif m['explained_var'] < 0.3:
            print(f"  ⚠️  Bajo (subí capacidad o vf_coef)")
        elif m['explained_var'] > 0.7:
            print(f"  ✅ Excelente")
        else:
            print(f"  ✅")
        
        # Entropy
        print(f"\n🎲 EXPLORATION:")
        print(f"   Entropy:    {m['entropy']:>8.4f}", end="")
        if m['entropy'] < 0.01:
            print(f"  ⚠️  MUY BAJO (política determinista)")
        elif m['entropy'] > 0.5:
            print(f"  💡 Alto (aún explorando)")
        else:
            print(f"  ✅")
        
        # Recomendaciones
        print(f"\n💡 RECOMENDACIONES:")
        recs = []
        if m['ratio_mean'] > 1.3 or abs(m['kl_div']) > 0.03:
            recs.append("   • Bajá LR (p.ej., 3e-4 → 1e-4)")
        if m['clip_frac'] > 0.6:
            recs.append("   • Reducí epochs (p.ej., 4 → 3)")
        if m['entropy'] < 0.05:
            recs.append("   • Subí ent_coef (p.ej., 0.01 → 0.02)")
        if m['explained_var'] < 0.2:
            recs.append("   • Subí vf_coef (0.5 → 1.0) o capacidad del crítico")
        if m['clip_frac'] < 0.03:
            recs.append("   • Podés subir LR (más agresivo)")
        
        if recs:
            for r in recs:
                print(r)
        else:
            print("   ✅ Todo se ve bien!")
        
        print(f"{'='*70}\n")

# Función helper para calcular explained variance
def explained_variance(y_pred, y_true):
    """Calcula la varianza explicada por el crítico"""
    var_y = np.var(y_true)
    return 1 - np.var(y_true - y_pred) / (var_y + 1e-8)
