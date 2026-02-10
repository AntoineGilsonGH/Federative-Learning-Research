"""
Training
"""


def train(server, nb_training_steps, honest_clients, byz_client):
    accuracy_history = []  # {"test":[], "validation":[]}

    for training_step in range(nb_training_steps):

        # Evaluate Global Model Every 10 Training Steps
        if training_step % 10 == 0:
            test_acc = server.compute_test_accuracy()
            accuracy_history.append(test_acc)
            print(f"--- Training Step {training_step}/{nb_training_steps} ---")
            print(f"Test Accuracy: {test_acc:.4f}")
        # Honest Clients Compute Gradients
        for client in honest_clients:
            client.compute_gradients()

        # Aggregate Honest Gradients
        honest_gradients = [
            client.get_flat_gradients_with_momentum() for client in honest_clients
        ]

        # Apply Byzantine Attack
        byz_vector = byz_client.apply_attack(honest_gradients)

        # Combine Honest and Byzantine Gradients
        gradients = honest_gradients + byz_vector

        # Update Global Model
        server.update_model_with_gradients(gradients)

        # Send (Updated) Server Model to Clients
        new_model = server.get_dict_parameters()
        for client in honest_clients:
            client.set_model_state(new_model)

    test_acc = server.compute_test_accuracy()
    accuracy_history.append(test_acc)
    print(f"--- Training Step {training_step}/{nb_training_steps} ---")
    print(f"Test Accuracy: {test_acc:.4f}")

    return accuracy_history
