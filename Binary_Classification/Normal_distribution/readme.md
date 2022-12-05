r"""Probability of a particular Fock basis state.

Computes the probability :math: $|\braket{\vec{n}|\psi}|^2$ of measuring
the given multi-mode Fock state based on the state :math: $\ket{\psi}$.

.. warning::

    Computing the Fock probabilities of states has exponential scaling
    in the Gaussian representation (for example states output by a
    Gaussian backend as a :class:`~.BaseGaussianState`).
    This shouldn't affect small-scale problems, where only a few Fock
    basis state probabilities need to be calculated, but will become
    evident in larger scale problems.

Args:
    n (Sequence[int]): the Fock state :math: $\ket{\vec{n}}$ that we want to measure the probability of

Keyword Args:
    cutoff (int): Specifies where to truncate the computation (default value is 10).
        Note that the cutoff argument only applies for Gaussian representation;
        states represented in the Fock basis will use their own internal cutoff dimension.

Returns:
    float: measurement probability
