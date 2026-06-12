import torch


def apply_controlled_unitary_to_state(
    state: torch.Tensor,
    U: torch.Tensor,
    n_qubits: int,
    control: int,
    target: int,
) -> torch.Tensor:
    """Apply controlled-U (control=|1>) on a dense state without building the 2^n x 2^n matrix.

    This assumes big-endian basis ordering where qubit 0 is the most significant bit.

    state: [1, 2**n, 1]
    U: [1,2,2] or [2,2]
    """
    if control == target:
        raise ValueError("control and target must be different")
    if target < 0 or target >= n_qubits or control < 0 or control >= n_qubits:
        raise ValueError("control/target out of range")
    if state.dim() != 3 or state.shape[-1] != 1:
        raise ValueError(f"Expected state shape [1, 2**n, 1], got {tuple(state.shape)}")

    g = U.squeeze(0) if U.dim() == 3 else U  # [2,2]

    # Move to a convenient shape: [...,2,2] for (control,target) tensor indices.
    psi = state.view(*([2] * n_qubits))

    # Bring control and target axes to the end (in order ... , control, target)
    psi_perm = psi.movedim([control, target], [-2, -1])
    # psi_perm: [2]* (n-2) + [2(control), 2(target)]

    # Split by control value.
    # If control==0: amplitudes unchanged.
    # If control==1: apply U on the target axis for each remaining index.
    ctrl1 = psi_perm[..., 1, :]  # [..., 2(target)]

    # Apply U on last axis (target)
    # ctrl1 is [..., 2(target)]
    # We want out [..., 2(target)] where out[j] = sum_k U[j,k]*ctrl1[k]
    ctrl1_out = torch.matmul(ctrl1, g.mT)  # [..., 2]

    psi_perm2 = psi_perm.clone()
    psi_perm2[..., 1, :] = ctrl1_out


    # Move axes back and reshape
    out = psi_perm2.movedim([-2, -1], [control, target])
    return out.reshape(1, 1 << n_qubits, 1)

