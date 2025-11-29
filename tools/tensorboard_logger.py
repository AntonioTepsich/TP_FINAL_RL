"""
 TensorBoard Logger para PPO Training
Registra todas las métricas importantes del entrenamiento
"""
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch
from datetime import datetime
import os


class TensorBoardLogger:
    """
    Logger completo para TensorBoard que registra:
    - Métricas de recompensas y episodios
    - Métricas de PPO (policy loss, value loss, entropy, etc.)
    - Hiperparámetros (learning rate, entropy coefficient)
    - Estadísticas de normalización
    - Métricas de rendimiento (velocidad, tiempos)
    """
    
    def __init__(self, log_dir=None, comment=""):
        """
        Args:
            log_dir: Directorio para los logs. Si es None, crea uno con timestamp
            comment: Comentario adicional para el nombre del directorio
        """
        if log_dir is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            import time
            microseconds = int((time.time() % 1) * 1000000)
            if comment:
                log_dir = f'runs/PPO_{timestamp}_{microseconds}_{comment}'
            else:
                log_dir = f'runs/PPO_{timestamp}_{microseconds}'

        self.writer = SummaryWriter(log_dir=log_dir)
        self.log_dir = log_dir
        print(f"TensorBoard iniciado: {log_dir}")
        print(f"   Para visualizar: tensorboard --logdir=runs")
    
    def log_episode_metrics(self, global_step, episode_rewards, episode_lengths=None):
        """
        Registra métricas relacionadas con episodios
        
        Args:
            global_step: Paso global actual
            episode_rewards: Lista de recompensas de episodios recientes
            episode_lengths: Lista opcional de longitudes de episodios
        """
        if not episode_rewards:
            return
        
        # Recompensas
        mean_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
        max_reward = np.max(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.max(episode_rewards)
        min_reward = np.min(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.min(episode_rewards)
        std_reward = np.std(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.std(episode_rewards)
        
        self.writer.add_scalar('Episode/Mean_Reward', mean_reward, global_step)
        self.writer.add_scalar('Episode/Max_Reward', max_reward, global_step)
        self.writer.add_scalar('Episode/Min_Reward', min_reward, global_step)
        self.writer.add_scalar('Episode/Std_Reward', std_reward, global_step)
        self.writer.add_scalar('Episode/Total_Episodes', len(episode_rewards), global_step)
        
        if episode_lengths:
            mean_length = np.mean(episode_lengths[-100:]) if len(episode_lengths) >= 100 else np.mean(episode_lengths)
            self.writer.add_scalar('Episode/Mean_Length', mean_length, global_step)
    
    def log_ppo_metrics(self, global_step, metrics):
        """
        Registra métricas de PPO del update
        
        Args:
            global_step: Paso global actual
            metrics: Dict con métricas de PPO (policy_loss, value_loss, entropy, etc.)
        """
        # Losses principales
        self.writer.add_scalar('PPO/Policy_Loss', metrics['policy_loss'], global_step)
        self.writer.add_scalar('PPO/Value_Loss', metrics['value_loss'], global_step)
        self.writer.add_scalar('PPO/Total_Loss', 
                             metrics['policy_loss'] + metrics['value_loss'], 
                             global_step)
        
        # Entropy
        self.writer.add_scalar('PPO/Entropy', metrics['entropy'], global_step)
        
        # KL Divergence
        self.writer.add_scalar('PPO/KL_Divergence', metrics['kl_div'], global_step)
        self.writer.add_scalar('PPO/Approx_KL', metrics['approx_kl'], global_step)
        
        # Clipping
        self.writer.add_scalar('PPO/Clip_Fraction', metrics['clip_frac'], global_step)
        
        # Ratio statistics
        self.writer.add_scalar('PPO/Ratio_Mean', metrics['ratio_mean'], global_step)
        self.writer.add_scalar('PPO/Ratio_Std', metrics['ratio_std'], global_step)
        
        # Explained Variance 
        self.writer.add_scalar('PPO/Explained_Variance', metrics['explained_var'], global_step)
        
        # Warnings como scalars binarios 
        self.writer.add_scalar('Warnings/High_KL', 1.0 if abs(metrics['kl_div']) > 0.03 else 0.0, global_step)
        self.writer.add_scalar('Warnings/Low_Explained_Var', 1.0 if metrics['explained_var'] < 0.2 else 0.0, global_step)
        self.writer.add_scalar('Warnings/Low_Entropy', 1.0 if metrics['entropy'] < 0.01 else 0.0, global_step)
    
    def log_hyperparameters(self, global_step, lr, entropy_coef, clip_eps=None, vf_coef=None):
        """
        Registra hiperparámetros que pueden cambiar durante el entrenamiento
        
        Args:
            global_step: Paso global actual
            lr: Learning rate actual
            entropy_coef: Coeficiente de entropía actual
            clip_eps: Epsilon de clipping (opcional)
            vf_coef: Coeficiente de value function (opcional)
        """
        self.writer.add_scalar('Hyperparameters/Learning_Rate', lr, global_step)
        self.writer.add_scalar('Hyperparameters/Entropy_Coefficient', entropy_coef, global_step)
        
        if clip_eps is not None:
            self.writer.add_scalar('Hyperparameters/Clip_Epsilon', clip_eps, global_step)
        if vf_coef is not None:
            self.writer.add_scalar('Hyperparameters/Value_Function_Coef', vf_coef, global_step)
    
    def log_performance(self, global_step, steps_per_sec, rollout_time=None, learn_time=None):
        """
        Registra métricas de rendimiento del entrenamiento
        
        Args:
            global_step: Paso global actual
            steps_per_sec: Pasos por segundo
            rollout_time: Tiempo de rollout (opcional)
            learn_time: Tiempo de aprendizaje (opcional)
        """
        self.writer.add_scalar('Performance/Steps_Per_Second', steps_per_sec, global_step)
        
        if rollout_time is not None:
            self.writer.add_scalar('Performance/Rollout_Time', rollout_time, global_step)
        if learn_time is not None:
            self.writer.add_scalar('Performance/Learn_Time', learn_time, global_step)
        if rollout_time is not None and learn_time is not None:
            total_time = rollout_time + learn_time
            self.writer.add_scalar('Performance/Total_Iteration_Time', total_time, global_step)
            self.writer.add_scalar('Performance/Rollout_Fraction', rollout_time / total_time, global_step)
    
    def log_normalization_stats(self, global_step, obs_mean, obs_var):
        """
        Registra estadísticas de normalización de observaciones
        
        Args:
            global_step: Paso global actual
            obs_mean: Media de observaciones
            obs_var: Varianza de observaciones
        """
        if isinstance(obs_mean, np.ndarray):
            self.writer.add_scalar('Normalization/Mean_Norm', np.linalg.norm(obs_mean), global_step)
            self.writer.add_scalar('Normalization/Var_Norm', np.linalg.norm(obs_var), global_step)
            
            self.writer.add_histogram('Normalization/Obs_Mean', obs_mean, global_step)
            self.writer.add_histogram('Normalization/Obs_Var', obs_var, global_step)
    
    def log_model_weights(self, global_step, model):
        """
        Registra histogramas de pesos y gradientes del modelo
        
        Args:
            global_step: Paso global actual
            model: Modelo PyTorch
        """
        for name, param in model.named_parameters():
            self.writer.add_histogram(f'Weights/{name}', param.data, global_step)
            if param.grad is not None:
                self.writer.add_histogram(f'Gradients/{name}', param.grad, global_step)
    
    def log_advantages_and_returns(self, global_step, advantages, returns):
        """
        Registra estadísticas de advantages y returns
        
        Args:
            global_step: Paso global actual
            advantages: Tensor de advantages
            returns: Tensor de returns
        """
        if isinstance(advantages, torch.Tensor):
            advantages = advantages.detach().cpu().numpy()
        if isinstance(returns, torch.Tensor):
            returns = returns.detach().cpu().numpy()
        
        self.writer.add_scalar('GAE/Advantages_Mean', np.mean(advantages), global_step)
        self.writer.add_scalar('GAE/Advantages_Std', np.std(advantages), global_step)
        self.writer.add_scalar('GAE/Returns_Mean', np.mean(returns), global_step)
        self.writer.add_scalar('GAE/Returns_Std', np.std(returns), global_step)
        
        self.writer.add_histogram('GAE/Advantages', advantages, global_step)
        self.writer.add_histogram('GAE/Returns', returns, global_step)
    
    def log_action_distribution(self, global_step, actions):
        """
        Registra la distribución de acciones tomadas (CRÍTICO para Flappy Bird)
        
        Args:
            global_step: Paso global actual
            actions: Array/Tensor de acciones tomadas en el rollout
        """
        if isinstance(actions, torch.Tensor):
            actions = actions.detach().cpu().numpy()
        
        if len(actions) == 0:
            return
        
        actions = actions.flatten()
        
        unique, counts = np.unique(actions, return_counts=True)
        
        for action in range(2):  
            freq = counts[unique == action][0] / len(actions) if action in unique else 0.0
            self.writer.add_scalar(f'Actions/Action_{action}_Frequency', freq, global_step)
        
        # Balance score 
        if len(unique) == 2:
            balance = min(counts) / max(counts)
            self.writer.add_scalar('Actions/Balance_Score', balance, global_step)
        else:
            # Si solo usa una acción, balance = 0
            self.writer.add_scalar('Actions/Balance_Score', 0.0, global_step)
        
        # Warning si colapsa a una sola acción
        collapsed = 1.0 if len(unique) == 1 else 0.0
        self.writer.add_scalar('Warnings/Action_Collapse', collapsed, global_step)
    
    def log_game_score(self, global_step, episode_scores):
        """
        Registra el score del juego (tubos pasados en Flappy Bird)
        
        Args:
            global_step: Paso global actual
            episode_scores: Lista de scores de episodios (tubos pasados)
        """
        if not episode_scores:
            return
        
        # Últimos 100 episodios o todos si hay menos
        scores = episode_scores[-100:] if len(episode_scores) >= 100 else episode_scores
        
        # Métricas básicas
        self.writer.add_scalar('Game/Mean_Pipes_Passed', np.mean(scores), global_step)
        self.writer.add_scalar('Game/Max_Pipes_Passed', np.max(scores), global_step)
        self.writer.add_scalar('Game/Best_Ever_Pipes', max(episode_scores), global_step)
        self.writer.add_scalar('Game/Std_Pipes_Passed', np.std(scores), global_step)
        
        # Distribución
        self.writer.add_histogram('Game/Pipes_Distribution', np.array(scores), global_step)
        
        # Success rate (pasó al menos 1 tubo)
        success_rate = sum(1 for s in scores if s > 0) / len(scores)
        self.writer.add_scalar('Game/Success_Rate', success_rate, global_step)
        
        # Percentiles
        if len(scores) >= 10:
            self.writer.add_scalar('Game/Pipes_Median', np.median(scores), global_step)
            self.writer.add_scalar('Game/Pipes_P75', np.percentile(scores, 75), global_step)
            self.writer.add_scalar('Game/Pipes_P90', np.percentile(scores, 90), global_step)
    
    def log_value_predictions(self, global_step, values, returns):
        """
        Analiza la calidad de las predicciones del crítico
        
        Args:
            global_step: Paso global actual
            values: Predicciones del crítico
            returns: Returns reales (targets)
        """
        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().numpy()
        if isinstance(returns, torch.Tensor):
            returns = returns.detach().cpu().numpy()
        
        # Flatten
        values = values.flatten()
        returns = returns.flatten()
        
        # Errores
        mse = np.mean((values - returns) ** 2)
        mae = np.mean(np.abs(values - returns))
        
        self.writer.add_scalar('Critic/MSE', mse, global_step)
        self.writer.add_scalar('Critic/MAE', mae, global_step)
        
        # Correlación (qué tan bien correlacionan predicciones con realidad)
        if len(values) > 1 and np.std(values) > 1e-8 and np.std(returns) > 1e-8:
            corr = np.corrcoef(values, returns)[0, 1]
            self.writer.add_scalar('Critic/Correlation', corr, global_step)
        
        # Estadísticas de valores predichos vs reales
        self.writer.add_scalar('Critic/Mean_Predicted_Value', np.mean(values), global_step)
        self.writer.add_scalar('Critic/Mean_Actual_Return', np.mean(returns), global_step)
        self.writer.add_scalar('Critic/Value_Std', np.std(values), global_step)
        
        # Warning si el error es muy alto
        self.writer.add_scalar('Warnings/High_Value_Error', 1.0 if mae > 10.0 else 0.0, global_step)
    
    def log_gradient_health(self, global_step, model):
        """
        Monitorea la salud de los gradientes (detecta explosion/vanishing)
        
        Args:
            global_step: Paso global actual
            model: Modelo PyTorch
        """
        total_norm = 0.0
        param_count = 0
        
        for name, p in model.named_parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2).item()
                total_norm += param_norm ** 2
                param_count += 1
                
                # Log norma por capa (solo cada 50 steps para no saturar)
                if global_step % 50 == 0:
                    self.writer.add_scalar(f'Gradients/{name}_norm', param_norm, global_step)
        
        if param_count > 0:
            total_norm = (total_norm ** 0.5)
            self.writer.add_scalar('Gradients/Total_Norm', total_norm, global_step)
            
            # Warnings
            self.writer.add_scalar('Warnings/Gradient_Explosion', 
                                  1.0 if total_norm > 10.0 else 0.0, global_step)
            self.writer.add_scalar('Warnings/Gradient_Vanishing', 
                                  1.0 if total_norm < 0.001 else 0.0, global_step)
    
    def log_text(self, tag, text, global_step=0):
        """
        Registra texto arbitrario
        
        Args:
            tag: Etiqueta del texto
            text: Texto a registrar
            global_step: Paso global (opcional)
        """
        self.writer.add_text(tag, text, global_step)
    
    def log_hparams(self, hparam_dict, metric_dict):
        """
        Registra hiperparámetros y métricas finales para comparación
        
        Args:
            hparam_dict: Dict de hiperparámetros (ej: {'lr': 3e-4, 'gamma': 0.99})
            metric_dict: Dict de métricas finales (ej: {'final_reward': 100.5})
        """
        self.writer.add_hparams(hparam_dict, metric_dict)
    
    def close(self):
        """Cierra el writer"""
        self.writer.flush()
        self.writer.close()
        print(f"📊 TensorBoard cerrado. Logs guardados en: {self.log_dir}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


if __name__ == "__main__":
    logger = TensorBoardLogger(comment="test")
    
    for step in range(100):
        logger.log_episode_metrics(
            global_step=step,
            episode_rewards=[10 + step * 0.5 + np.random.randn() for _ in range(10)]
        )
        
        # Métricas de PPO
        metrics = {
            'policy_loss': 0.1 - step * 0.001,
            'value_loss': 0.5 - step * 0.003,
            'entropy': 0.5 - step * 0.004,
            'kl_div': 0.01 + np.random.randn() * 0.005,
            'approx_kl': 0.01 + np.random.randn() * 0.005,
            'clip_frac': 0.1 + np.random.rand() * 0.1,
            'ratio_mean': 1.0 + np.random.randn() * 0.1,
            'ratio_std': 0.2 + np.random.rand() * 0.05,
            'explained_var': 0.8 + np.random.rand() * 0.15
        }
        logger.log_ppo_metrics(step, metrics)
        
        # Hiperparámetros
        logger.log_hyperparameters(
            global_step=step,
            lr=3e-4 * (1 - step / 100),
            entropy_coef=0.01
        )
        
        # Performance
        logger.log_performance(
            global_step=step,
            steps_per_sec=1000 + step * 10,
            rollout_time=0.5,
            learn_time=0.3
        )
    
    logger.close()
    print(" Test completado. Ejecuta: tensorboard --logdir=runs")
