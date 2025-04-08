def optimize_model(self, experience):
    states, actions, rewards, next_states, is_terminals = experience
    batch_size = len(is_terminals)

    max_a_q_sp = self.target_model(next_states).detach().max(1)[0].unsqueeze(1)
    target_q_sa = rewards + (self.gamma * max_a_q_sp * (1 - is_terminals))

    q_sa = self.online_model(states).gather(1, actions)

    td_error = q_sa - target_q_sa
    value_loss = td_error.pow(2).mul(0.5).mean()
    self.value_optimizer.zero_grad()
    value_loss.backward()
    self.value_optimizer.step()


def interation_step(self, state, env):
    action = self.training_strategy.select_action(self.online_model, state)
    new_state, reward, is_terminal, info = env.step(action)
