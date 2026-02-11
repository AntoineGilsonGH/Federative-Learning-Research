"""
Training
"""
import torch

def train_attacks(server, nb_training_steps, honest_clients, byz_client):
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
        byz_updates = byz_client.apply_attack(honest_gradients)

        # byz_updates peut être un tensor OU une liste de tensors
        if not isinstance(byz_updates, list):
            byz_updates = [byz_updates]

        ref = honest_gradients[0]  # pour device + dtype de référence
        fixed_byz = []
        for u in byz_updates:
            # si c'est du numpy, convertit en torch
            if not torch.is_tensor(u):
                u = torch.tensor(u)

            # aligne device + dtype
            u = u.to(device=ref.device, dtype=ref.dtype)
            fixed_byz.append(u)

        gradients = honest_gradients + fixed_byz


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
