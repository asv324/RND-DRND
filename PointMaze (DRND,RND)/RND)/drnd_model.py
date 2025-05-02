import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch.nn import init

class MlpNetwork(nn.Module):
    """
    Basic MLP network for policy and value functions
    """
    def __init__(self, input_dim, output_dim, hidden_dim=256, activation=nn.ReLU):
        super(MlpNetwork, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, output_dim)
        
        self.activation = activation()
        
        # Initialize weights with orthogonal initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.orthogonal_(m.weight, np.sqrt(2))
                m.bias.data.zero_()
    
    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        return x

class ActorCriticNetwork(nn.Module):
    """
    Actor-Critic network for PPO with multiple value heads
    """
    def __init__(self, input_size, action_size, std_init=0.5):
        super(ActorCriticNetwork, self).__init__()
        
        # Shared feature extractor
        self.feature = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        
        # Actor (policy) network
        self.actor_mean = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_size)
        )
        self.actor_log_std = nn.Parameter(torch.ones(1, action_size) * np.log(std_init))
        
        # Extra layer for value functions
        self.extra_layer = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU()
        )
        
        # Value networks - separate heads for extrinsic and intrinsic rewards
        self.critic_ext = nn.Linear(256, 1)  # Extrinsic value
        self.critic_int = nn.Linear(256, 1)  # Intrinsic value
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for p in self.modules():
            if isinstance(p, nn.Conv2d):
                init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()
            
            if isinstance(p, nn.Linear):
                init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()
        
        # Special initialization for output layers
        for i in range(len(self.actor_mean)):
            if isinstance(self.actor_mean[i], nn.Linear):
                init.orthogonal_(self.actor_mean[i].weight, 0.01)
                self.actor_mean[i].bias.data.zero_()
        
        init.orthogonal_(self.critic_ext.weight, 0.01)
        self.critic_ext.bias.data.zero_()
        
        init.orthogonal_(self.critic_int.weight, 0.01)
        self.critic_int.bias.data.zero_()
        
        for i in range(len(self.extra_layer)):
            if isinstance(self.extra_layer[i], nn.Linear):
                init.orthogonal_(self.extra_layer[i].weight, 0.1)
                self.extra_layer[i].bias.data.zero_()
    
    def forward(self, state):
        """
        Forward pass with NaN detection and prevention
        """
        # Check for NaN in input
        if torch.isnan(state).any() or torch.isinf(state).any():
            print("WARNING: NaN/Inf values detected in network input")
            state = torch.nan_to_num(state, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Extract shared features
        x = self.feature(state)
        
        # Actor outputs
        action_mean = self.actor_mean(x)
        action_log_std = self.actor_log_std.expand_as(action_mean)
        
        # Critic outputs using shared features and extra layer
        extra = self.extra_layer(x)
        value_ext = self.critic_ext(extra)
        value_int = self.critic_int(extra)
        
        return action_mean, action_log_std, value_ext, value_int


class DRNDModel(nn.Module):
    """
    Distributional Random Network Distillation model
    """
    def __init__(self, input_size, output_dim=512, num_targets=10, device="cuda" if torch.cuda.is_available() else "cpu"):
        super(DRNDModel, self).__init__()
        
        self.input_size = input_size
        self.output_dim = output_dim
        self.num_targets = num_targets
        self.device = device
        
        # Predictor network
        self.predictor = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, output_dim)
        ).to(device)
        
        # Multiple target networks for distributional approach
        self.targets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_size, 256),
                nn.LeakyReLU(),
                nn.Linear(256, 256),
                nn.LeakyReLU(),
                nn.Linear(256, output_dim)
            ).to(device) for _ in range(num_targets)
        ])
        
        # Initialize weights using orthogonal initialization
        self._init_weights()
        
        # Freeze target networks - they don't get updated
        for target_net in self.targets:
            for param in target_net.parameters():
                param.requires_grad = False
    
    def _init_weights(self):
        """Initialize network weights"""
        for p in self.modules():
            if isinstance(p, nn.Conv2d):
                init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()
            
            if isinstance(p, nn.Linear):
                init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()
    
    def forward(self, obs):
        """
        Forward pass through predictor and target networks
        
        Args:
            obs: Observation tensor
            
        Returns:
            predict_feature: Output from predictor network
            target_features: Outputs from all target networks
        """
        # Run observation through predictor network
        predict_feature = self.predictor(obs)
        
        # Run observation through all target networks
        target_features = torch.zeros(self.num_targets, obs.shape[0], self.output_dim, device=obs.device)
        for i, target_net in enumerate(self.targets):
            target_features[i] = target_net(obs)
        
        return predict_feature, target_features

class DRNDAgent:
    """
    Agent implementing DRND exploration for PPO
    """
    def __init__(
        self,
        input_size,
        action_size,
        device="cuda" if torch.cuda.is_available() else "cpu",
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        ppo_epsilon=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        num_targets=10,
        drnd_output_dim=512,
        alpha=0.9,  # Mixing coefficient for DRND bonus terms
        drnd_lr=3e-4
    ):
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ppo_epsilon = ppo_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.alpha = alpha
        
        # Initialize actor-critic model
        self.model = ActorCriticNetwork(input_size, action_size).to(device)
        
        # Initialize DRND model
        self.drnd = DRNDModel(
            input_size=input_size,
            output_dim=drnd_output_dim,
            num_targets=num_targets,
            device=device
        ).to(device)
        
        # Setup optimizers
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.drnd_optimizer = torch.optim.Adam(self.drnd.predictor.parameters(), lr=drnd_lr)
    
    def get_action(self, state):
        """
        Get action from policy with exploration
        
        Args:
            state: Current state tensor
            
        Returns:
            action: Sampled action
            action_log_prob: Log probability of the action
            value_ext: Extrinsic value estimate
            value_int: Intrinsic value estimate
            entropy: Entropy of the policy
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).to(self.device)
            action_mean, action_log_std, value_ext, value_int = self.model(state_tensor)
            
            # Create normal distribution with predicted mean and std
            std = torch.exp(action_log_std)
            dist = torch.distributions.Normal(action_mean, std)
            
            # Sample action from the distribution
            action = dist.sample()
            action_log_prob = dist.log_prob(action).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1).mean()
            
            # Clip actions to [-1, 1] for environments with bounded action space
            action = torch.tanh(action)
            
            return (
                action.cpu().numpy(),
                action_log_prob.cpu().numpy(),
                value_ext.cpu().numpy().squeeze(),
                value_int.cpu().numpy().squeeze(),
                entropy.cpu().numpy(),
                (action_mean.cpu().numpy(), action_log_std.cpu().numpy())  # Store policy params for PPO
            )
    
    def compute_intrinsic_reward(self, states):
        """
        Compute intrinsic reward using DRND
        
        Args:
            states: Batch of states
            
        Returns:
            intrinsic_rewards: DRND-based intrinsic rewards
        """
        with torch.no_grad():
            if not isinstance(states, torch.Tensor):
                states = torch.FloatTensor(states).to(self.device)
            
            # Get features from predictor and target networks
            predict_feature, target_features = self.drnd(states)
            
            # Compute mean and second moment of target networks' outputs
            mu = torch.mean(target_features, dim=0)  # Mean across target networks
            B2 = torch.mean(target_features**2, dim=0)  # Second moment
            
            # First bonus term: MSE between predictor and mean of targets
            bonus1 = ((predict_feature - mu)**2).sum(dim=1)
            
            # Second bonus term: Pseudo-count estimation 
            # Following the formula from the paper
            epsilon = 1e-8  # Small constant to avoid division by zero
            numerator = torch.clamp(predict_feature**2 - mu**2, min=0)
            denominator = torch.clamp(B2 - mu**2, min=epsilon)
            pseudo_counts = torch.sqrt(numerator / denominator).sum(dim=1)
            
            # Combine both bonuses with alpha as the mixing coefficient
            intrinsic_rewards = self.alpha * bonus1 + (1 - self.alpha) * pseudo_counts
            
            return intrinsic_rewards.cpu().numpy()
    
    def update_drnd(self, states, update_proportion=0.25):
        """
        Update the DRND predictor network
        
        Args:
            states: Batch of states
            update_proportion: Proportion of states to use for updating
            
        Returns:
            loss: DRND prediction loss
        """
        if not isinstance(states, torch.Tensor):
            states = torch.FloatTensor(states).to(self.device)
        
        # Get features from predictor and target networks
        predict_feature, target_features = self.drnd(states)
        
        # Sample a target network for each state
        batch_size = states.shape[0]
        indices = torch.randint(0, self.drnd.num_targets, (batch_size,), device=self.device)
        
        # Get the corresponding target for each state
        selected_targets = target_features[indices, torch.arange(batch_size)]
        
        # Calculate MSE loss
        loss = F.mse_loss(predict_feature, selected_targets, reduction='none').mean(dim=1)
        
        # Only update for a random subset of states to maintain exploration
        mask = torch.rand(batch_size, device=self.device) < update_proportion
        mask = mask.float()
        mask_sum = mask.sum() + 1e-8  # Avoid division by zero
        masked_loss = (loss * mask).sum() / mask_sum
        
        # Update predictor network
        self.drnd_optimizer.zero_grad()
        masked_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.drnd.predictor.parameters(), self.max_grad_norm)
        self.drnd_optimizer.step()
        
        return masked_loss.item()
    
    def update_policy(self, states, actions, old_log_probs, returns_ext, returns_int, advantages):
        """
        Update policy and value networks using PPO
        
        Args:
            states: Batch of states
            actions: Batch of actions taken
            old_log_probs: Log probabilities of actions under old policy
            returns_ext: Extrinsic returns
            returns_int: Intrinsic returns
            advantages: Combined advantages (extrinsic + intrinsic)
            
        Returns:
            policy_loss: Actor loss
            value_loss: Critic loss
            entropy: Policy entropy
        """
        if not isinstance(states, torch.Tensor):
            states = torch.FloatTensor(states).to(self.device)
        if not isinstance(actions, torch.Tensor):
            actions = torch.FloatTensor(actions).to(self.device)
        if not isinstance(old_log_probs, torch.Tensor):
            old_log_probs = torch.FloatTensor(old_log_probs).to(self.device)
        if not isinstance(returns_ext, torch.Tensor):
            returns_ext = torch.FloatTensor(returns_ext).to(self.device)
        if not isinstance(returns_int, torch.Tensor):
            returns_int = torch.FloatTensor(returns_int).to(self.device)
        if not isinstance(advantages, torch.Tensor):
            advantages = torch.FloatTensor(advantages).to(self.device)
        
        # Get current policy outputs
        action_mean, action_log_std, value_ext, value_int = self.model(states)
        
        # Calculate current action probabilities
        std = torch.exp(action_log_std)
        dist = torch.distributions.Normal(action_mean, std)
        
        # Get log probs of actions under current policy
        curr_log_probs = dist.log_prob(actions).sum(dim=1)
        
        # Calculate policy ratio and clipped surrogate objective
        ratio = torch.exp(curr_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.ppo_epsilon, 1.0 + self.ppo_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Calculate value loss for extrinsic and intrinsic returns
        value_ext_loss = F.mse_loss(value_ext.squeeze(), returns_ext)
        value_int_loss = F.mse_loss(value_int.squeeze(), returns_int)
        value_loss = value_ext_loss + value_int_loss
        
        # Calculate entropy bonus
        entropy = dist.entropy().sum(dim=1).mean()
        
        # Total loss
        total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
        
        # Update networks
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        
        return policy_loss.item(), value_loss.item(), entropy.item()