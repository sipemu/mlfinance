//! Represent a trajectory as a sequence of states.
//! Equivalent to Snippets 21.2-21.3.

/// Generate all possible trajectories of a given length from an initial state.
///
/// At each step, the trajectory branches by adding every element in `moves`
/// to the current state, producing a full tree of paths.
/// Equivalent to Snippets 21.2-21.3.
///
/// # Arguments
///
/// * `initial_state` - Starting value for every trajectory.
/// * `moves` - Possible increments at each step.
/// * `steps` - Number of steps (each trajectory will have `steps + 1` states).
///
/// # Returns
///
/// A vector of trajectories. The count equals `moves.len().pow(steps)`.
pub fn generate_trajectories(initial_state: f64, moves: &[f64], steps: usize) -> Vec<Vec<f64>> {
    let mut trajectories: Vec<Vec<f64>> = vec![vec![initial_state]];

    for _ in 0..steps {
        let mut new_trajectories = Vec::new();
        for traj in &trajectories {
            let last = *traj.last().unwrap();
            for &m in moves {
                let mut new_traj = traj.clone();
                new_traj.push(last + m);
                new_trajectories.push(new_traj);
            }
        }
        trajectories = new_trajectories;
    }

    trajectories
}

/// Evaluate a payoff function on every trajectory.
///
/// # Arguments
///
/// * `trajectories` - Slice of trajectory paths (each is a `Vec<f64>`).
/// * `payoff` - Function mapping a trajectory to a scalar payoff.
///
/// # Returns
///
/// A vector of payoff values, one per trajectory.
pub fn evaluate_trajectories<F>(trajectories: &[Vec<f64>], payoff: F) -> Vec<f64>
where
    F: Fn(&[f64]) -> f64,
{
    trajectories.iter().map(|t| payoff(t)).collect()
}

/// Compute the expected payoff under uniform probability across trajectories.
///
/// # Arguments
///
/// * `trajectories` - Slice of trajectory paths.
/// * `payoff` - Function mapping a trajectory to a scalar payoff.
///
/// # Returns
///
/// The arithmetic mean of the payoff values. Returns 0.0 when `trajectories` is empty.
pub fn expected_value<F>(trajectories: &[Vec<f64>], payoff: F) -> f64
where
    F: Fn(&[f64]) -> f64,
{
    if trajectories.is_empty() {
        return 0.0;
    }
    let values = evaluate_trajectories(trajectories, payoff);
    values.iter().sum::<f64>() / values.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_trajectories() {
        // Binary tree: +1 or -1 at each step
        let trajs = generate_trajectories(0.0, &[1.0, -1.0], 3);
        assert_eq!(trajs.len(), 8); // 2^3
        assert_eq!(trajs[0].len(), 4); // initial + 3 steps
    }

    #[test]
    fn test_expected_value() {
        let trajs = generate_trajectories(0.0, &[1.0, -1.0], 2);
        // Terminal values: 2, 0, 0, -2; mean = 0
        let ev = expected_value(&trajs, |t| *t.last().unwrap());
        assert!((ev).abs() < 1e-10);
    }
}
