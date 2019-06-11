---
layout: post
title:  Conditional Distributions
categories: notebooks
---

Gaussian filters use statistical linearization to obtain mean and covariance estimates. Suppose that $x \sim N(\pmb{x_0}, P_x)$ and
\begin{equation}
\pmb{y} = f(\pmb{x}) 
\end{equation}
where $f : \mathbb{R}^n \to \mathbb{R}^m$ is a nonlinear function. According to {% cite Sarkka2013 %}, in statistical linearization, one forms a linear approximation of the nonlinear transformation 
\begin{equation}
g(\pmb{x}) \approx \pmb{b} + A \delta \pmb{x} = \pmb{b} + A (\pmb{x} - \pmb{x_0})
\end{equation}
that minimizes the mean square error
\begin{equation}
\text{MSE}(\pmb{b}, A) = E[(f(\pmb{x}) - \pmb{b} -  A \delta \pmb{x})\; (f(\pmb{x}) - \pmb{b} -  A \delta \pmb{x})^T].
\end{equation}
Setting derivatives with respect to $A$ and $\pmb{b}$ yields
\begin{equation}
\pmb{b} = E[f(\pmb{x})]
\end{equation}
\begin{equation}
A = E[f(\pmb{x}) \delta \pmb{x}] P_x^{-1}.
\end{equation}
Here  $\pmb{b} = E[f(\pmb{x})]$ is exactly the mean of $\pmb{y} = f(\pmb{x})$. The approximate covariance is given by 
\begin{equation}
\begin{gathered}
P_y = E[(f(\pmb{x}) - E(f(\pmb{x})) \; (f(\pmb{x}) - E(f(\pmb{x}))^T] \\\ 
\approx A P_x A^T = E[f(\pmb{x}) \; \delta \pmb{x}^T] P_x^{-1} E[f(\pmb{x}) \; \delta \pmb{x}^T].
\end{gathered}
\end{equation}
Note that statistical linearization involves computing expectation integrals $E[f(\pmb{x})]$ and $E[f(\pmb{x}) \; \delta \pmb{x}^T]$ of the form discussed in the sections on [Gaussian quadrature]({% post_url sigmapy/2019-06-10-gaussian_quadrature %}).


## Conditional Distribution for an Additive Transform 

Consider a noisy additive transform of the form 
\begin{equation}
\begin{gathered}
\pmb{y} = f(\pmb{x}) + \pmb{q} \\\
x \sim N(\pmb{x_0}, P_x) \\\
q \sim N(\pmb{0}, Q)
\end{gathered}
\end{equation}
where $\pmb{q}$ is the measurement noise. Suppose we have some observation $\pmb{y_0}$ and want to compute the mean and covariance of the probability distribution 
\begin{equation}
P( \pmb{x} | \pmb{y_0}).
\end{equation}
Performing statistical linearization on the augmented function $\hat{f}(\pmb{x}) = [\pmb{x}, f(\pmb{x})]$ yields the following approximations for mean and covariance
\begin{equation}
E[\hat{f}(\pmb{x})] \approx
\begin{bmatrix}
\pmb{x_0} \\\
E[f(\pmb{x})]
\end{bmatrix}
=
\begin{bmatrix}
\pmb{x_0} \\\
\pmb{\mu}
\end{bmatrix}
\end{equation}

\begin{equation}
\begin{gathered}
\text{Cov}[\hat{f}(\pmb{x})] \approx
\begin{bmatrix}
P_x & E[f(\pmb{x}) \\; \delta \pmb{x}^T]^T \\\
E[f(\pmb{x}) \\; \delta \pmb{x}^T] & E[f(\pmb{x}) \\; \delta \pmb{x}^T] P_x^{-1} E[f(\pmb{x}) \\; \delta \pmb{x}^T] + Q
\end{bmatrix} \\\
= 
\begin{bmatrix}
 P_x & C \\\
 C^T & S 
\end{bmatrix}
\end{gathered}
\end{equation}
Here, $\mu$ is the measurement mean, and $S$ and $C$ are the measurement covariance and cross covariance respectively. Stated otherwise, statistical linearization yields a Gaussian approximation of the joint distribution 
\begin{equation}
\begin{bmatrix}
\pmb{x} \\\
\pmb{y}
\end{bmatrix}
\sim 
N \left ( \begin{bmatrix}
\pmb{x_0} \\\
\pmb{\mu}
\end{bmatrix},  \begin{bmatrix}
 P_x & C \\\
 C^T & S
\end{bmatrix}\right ).
\end{equation}
Given tge measurement $\pmb{y_0}$, the mean and covariance of $\pmb{x} | \pmb{y_0}$ can be computed from the joint distribution as follows
\begin{equation}
\pmb{x'}  \sim N \left ( \pmb{x_0} + K[\pmb{y_o} - \pmb{\mu}], \\; P_x - K S K^T \right )
\end{equation}
where $K = CS^{-1}$ is the Kalman gain.

## Bibliography
{% bibliography --cited %}

