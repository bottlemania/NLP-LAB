# Purpose: This program implements the forward and backward algorithms for a Hidden Markov Model (HMM).

def main():

    # Define the state space
    states = ["CP","IP"]
    n_states = len(states)

    # Define the observation space
    observations = ["a", "b", "c"]
    n_observations = len(observations)

    # Define the initial state distribution
    pi = [1.0, 0.0]  

    # Define the state transition probabilities
    A = [[0.7, 0.3],
        [0.5, 0.5]]

    # Define the observation likelihoods
    B = [[0.6, 0.1, 0.3],
        [0.1, 0.7, 0.2]]

    # required output sequence
    op_seq = "c b a".split(" ")
    n_op = len(op_seq)
    t = n_op + 1

    alpha, prob_alpha = forward_algorithm(pi, A, B, op_seq, n_states, t, observations)
    beta, prob_beta = backward_algorithm(pi, A, B, op_seq, n_states, t, observations)

    print("------------- Forward procedure -------------\n")
    for index,row in enumerate(alpha):
        row=[round(x,6) for x in row]
        print("alpha[{}]: ".format(states[index]), row) 
    print("Prob_array: ", [round(x,6) for x in prob_alpha])
    print("Probability of the observation sequence: ", round(prob_alpha[-1],6))

    print("\n------------- Backward procedure -------------\n")
    for index,row in enumerate(beta):
        row=[round(x,6) for x in row]
        print("beta[{}]: ".format(states[index]), row)

    print("Probability of the observation sequence: ", round(prob_beta,6))



def forward_algorithm(pi, A, B, op_seq, n_states, t, observations):

    alpha = [[0] * t for i in range(n_states)]  # Initialize alpha
    prob = []

    # Initialize the forward algorithm
    for i in range(n_states):
        alpha[i][0] = pi[i]

    # Execute the forward algorithm
    for time in range(1, t):
        obv = observations.index(op_seq[time - 1])
        for j in range(n_states):
            alpha[j][time] = 0
            for i in range(n_states):
                alpha[j][time] += alpha[i][time - 1] * A[i][j] * B[i][obv]

    # Calculate the probability of the observation sequence
    for i in range(t):
        prob.append(sum(alpha[j][i] for j in range(n_states)))    
    
    return alpha, prob



def backward_algorithm(pi, A, B, op_seq, n_states, t, observations):

    beta = [[0] * t for j in range(n_states)] # Initialize beta
    prob = 0

    # Initialize the backward algorithm
    for i in range(n_states):
        beta[i][-1] = 1

    # Execute the backward algorithm
    for time in range(t - 2, -1, -1):
        obv = observations.index(op_seq[time])
        for i in range(n_states):
            beta[i][time] = 0
            for j in range(n_states):
                beta[i][time] += beta[j][time + 1] * A[i][j] * B[i][obv]

    # Calculate the probability of the observation sequence
    prob=+(sum((pi[i]*beta[i][0]) for i in range(n_states)))

    return beta, prob


if __name__ == "__main__":
    main()