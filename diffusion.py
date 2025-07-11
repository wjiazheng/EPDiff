
"""
This code is based on @dome272 implementation of DDPM's
https://github.com/dome272/Diffusion-Models-pytorch
"""

import yaml
from utils_eval_brats import *
import cv2

class Diffusion:
    def __init__(self, noise_steps, img_size, beta_start, beta_end, device):
        self.noise_steps = noise_steps
        
        self.beta = self.linear_noise_schedule(beta_start, beta_end).to(device)
        self.alpha = 1.0 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        self.img_size = img_size
        self.device = device

    def linear_noise_schedule(self, beta_start, beta_end):
        return torch.linspace(beta_start, beta_end, self.noise_steps)

    def noise_images(self, x, t, pyramid=False, discount = 0.8):
        """This method generates the latent representations at time t of the input images. Equation (1) in ANDi paper

        Parameters
        ----------
        x : tensor
            Input x_0
        t : tensor
            tensor containing the values for the time steps t
        pyramid : bool, optional
            flag that decides if pyramid noise is used, by default False
        discount : float, optional
            the discount for the pyramid noise, by default 0.8

        Returns
        -------
        tensor, tensor
            The first tensor contains the latent representations, the second the noise used (needed for training)
        """
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[
            :, None, None, None
        ]
        if pyramid == True:
                noise = pyramid_noise_like(x.shape[0], x.shape[1], self.img_size, discount, self.device)

        else:
            noise = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * noise, noise

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def ddpm_mu_t(self, x, predicted_noise, t):
        """This method calculates the mean of the Gaussian transition at time t given the noise. ANDi paper notation mu_q, or mu_theta.

        Parameters
        ----------
        x : tensor
            The tensor containing x_t
        predicted_noise : tensor
            The tensor containing the noise (either predicted or ground truth)
        t : tensor
            The tensor containing the time steps

        Returns
        -------
        tensor
            The tensor contatining x_t-1
        """
        alpha = self.alpha[t][:, None, None, None]
        alpha_hat = self.alpha_hat[t][:, None, None, None]
        return (
            1
            / torch.sqrt(alpha)
            * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise)
        )

    def ddpm_mean_t(self, x, t, predicted_noise=None, x_0=None):
        """This method calculates the mean of the Gaussian transition at time t given the predicted noise, or the original sample x_0.
        Additionally clips the predictions like the original implementation of DDPM's, when the noise is provided to calculate the mean.

        Parameters
        ----------
        x : tensor
            The tensor containing x_t
        predicted_noise : tensor
            The tensor containing the noise (either predicted or ground truth)
        t : tensor
            The tensor containing the time steps
        x_0 : tensor
            The tensor containing x_0

        Returns
        -------
        tensor
            The tensor contatining x_t-1
        """
        if predicted_noise == None and x_0 == None:
            print("Either noise or x_0 have to be given to calculate x_t-1.")
            exit(1)
        alpha = self.alpha[t][:, None, None, None]
        alpha_hat = self.alpha_hat[t][:, None, None, None]
        beta = self.beta[t][:, None, None, None]
        alpha_hat_minus_one = self.alpha_hat[t - 1][:, None, None, None]
        if x_0 == None:
            pred_x0 = (
                1 / torch.sqrt(alpha_hat) * x
                - torch.sqrt((1 - alpha_hat) / (alpha_hat)) * predicted_noise
            )
            pred_x0 = pred_x0.clamp(-1, 1)
            x_0 = pred_x0
        w0 = torch.sqrt(alpha_hat_minus_one) * beta / (1 - alpha_hat)
        wt = torch.sqrt(alpha) * (1 - alpha_hat_minus_one) / (1 - alpha_hat)
        return w0 * x_0 + wt * x

    def sample(
        self, model, n, channels, pyramid=False, discount = 0.8
    ):
        """This method samples from the learned DDPM.

        Parameters
        ----------
        model : _type_
            The learned DDPM model (U-Net)
        n : int
            The number of samples you want to create
        channels : int
            The number of channels the model is trained on. For ANDi, this is 4 (FLAIR, T1, T1ce, T2)
        pyramid : bool, optional
            flag that decides if pyramid noise is used, by default False
        discount : float, optional
            the discount for the pyramid noise, by default 0.8

        Returns
        -------
        tensor
            The samples created.
        """
        model.eval()
        with torch.no_grad():
            if pyramid == True:
                x = pyramid_noise_like(n, channels, self.img_size, discount, self.device)
            else:
                x = torch.randn((n, channels, self.img_size, self.img_size)).to(
                    self.device
                )
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t)
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                alpha_hat_minus_one = self.alpha_hat[t - 1][:, None, None, None]
                if i > 1:
                    if pyramid == True:
                        noise = pyramid_noise_like(n, channels, self.img_size, discount, self.device)
                    else:
                        noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                var = beta * (1 - alpha_hat_minus_one) / (1 - alpha_hat)
                x = self.ddpm_mu_t(x, predicted_noise, t) + torch.sqrt(var) * noise
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x

    def normative_diffusion(self, model, images, k=0, start=75, stop=200, pyramid=False, discount = 0.8, visualization = None):
        """This method calculates the deviations for each time step t in the interval T_l, T_u.
        In the ANDi paper this method corresponds to line 3 to 9 in the pseudo-code.

        Parameters
        ----------
        model : _type_
            The learned DDPM model (U-Net)
        images : tensor
            The samples x_0
        start : int, optional
            Lower endpoint of interval T_l, by default 75
        stop : int, optional
            Upper endpoint of interval T_u, by default 200
        pyramid : bool, optional
            flag that decides if pyramid noise is used, by default False
        discount : float, optional
            the discount for the pyramid noise, by default 0.8

        Returns
        -------
        tensor
            The tensor that contains the deviations for each time step in the dimension with the index 1
        """
        if stop == None:
            stop = self.noise_steps
        if start == 0:  # The start can not be the original sample x_0
            start = 1
        num_images = images.shape[0]
        model.eval()
        with torch.no_grad():
            dts = torch.zeros(
                (
                    num_images,
                    stop - start,
                    images.shape[1],
                    images.shape[2],
                    images.shape[3],
                )
            ).to(self.device)
            md = 0
            # calculate the deviations
            for i in tqdm(reversed(range(start, stop)), position=0):
                t = (torch.ones(num_images) * i).long().to(self.device)
                x_t, noise = self.noise_images(images, t, pyramid=pyramid, discount=discount)
                predicted_noise = model(x_t, t)
                mu_theta = self.ddpm_mu_t(x_t, predicted_noise, t)
                mu_q = self.ddpm_mu_t(x_t, noise, t)
                d_t = (mu_q - mu_theta) ** 2
                dts[:, i - start] = d_t
                if "visual" in visualization:
                    mu_theta2 = self.normlize(mu_theta.cpu().numpy())
                    mu_q2 = self.normlize(mu_q.cpu().numpy())
                    d_t2 = self.normlize(d_t.cpu().numpy())
                    cv2.imwrite(visualization+'/'+str(k)+"_"+str(i)+'_140_'+str(md)+'_predict.png',mu_theta2[140,md,:,:])
                    cv2.imwrite(visualization + '/' +str(k)+"_"+ str(i) + '_140_'+str(md)+'_gt.png', mu_q2[140, md, :, :])
                    cv2.imwrite(visualization + '/' +str(k)+"_"+ str(i) + '_140_'+str(md)+'_delta.png', d_t2[140, md, :, :])
                    cv2.imwrite(visualization + '/' + str(k) + "_" + str(i) + '_115_'+str(md)+'_predict.png', mu_theta2[115, md, :, :])
                    cv2.imwrite(visualization + '/' + str(k) + "_" + str(i) + '_115_'+str(md)+'_gt.png', mu_q2[115, md, :, :])
                    cv2.imwrite(visualization + '/' + str(k) + "_" + str(i) + '_115_'+str(md)+'_delta.png', d_t2[115, md, :, :])
                    if i == 1:
                        for j in range(0, mu_q2.shape[0]):
                            cv2.imwrite(visualization + '/' + str(k) + "_1_" + str(j) + '_'+str(md)+'_gt.png', mu_q2[j, md, :, :])
                            cv2.imwrite(visualization + '/' + str(k) + "_1_" + str(j) + '_'+str(md)+'_delta.png', d_t2[j, md, :, :])
                            cv2.imwrite(visualization + '/' + str(k) + "_1_" + str(j) + '_'+str(md)+'_predict.png',mu_theta2[j, md, :, :])
        return dts

    def normlize(self, img):
        img = 255 * (img-img.min())/(img.max()-img.min())
        return img

    def normative_blocks(
        self, model, images, start=75, stop=200, skip=25, pyramid=False, discount = 0.8
    ):
        """Experimental method. Not described in ANDi paper.
        It calculates deviations for blocks of the diffusion markov chain with setting the variance to zero.

        Parameters
        ----------
        model : _type_
            The learned DDPM model (U-Net)
        images : tensor
            The samples x_0
        start : int, optional
            Lower endpoint of interval T_l, by default 75
        stop : int, optional
            Upper endpoint of interval T_u, by default 200
        skip : int, optional
            The number of deviations to skip before calculating the deviation for a block, by default 25
        pyramid : bool, optional
            flag that decides if pyramid noise is used, by default False
        discount : float, optional
            the discount for the pyramid noise, by default 0.8

        Returns
        -------
        tensor
            The tensor that contains the deviations for each block in the dimension with the index 1
        """
        if stop is None:
            stop = self.noise_steps
        if start == 0:  # The start can not be the original sample x_0
            start = 1
        num_images = images.shape[0]
        model.eval()
        with torch.no_grad():
            dts = torch.zeros(
                (
                    num_images,
                    int(((stop - start) / skip)),
                    images.shape[1],
                    images.shape[2],
                    images.shape[3],
                )
            ).to(self.device)

            t = (torch.ones(num_images) * stop - 1).long().to(self.device)
            x_t, noise = self.noise_images(images, t, pyramid=pyramid, discount = discount)
            correct_chain = x_t
            predicted_chain = x_t

            for i in tqdm(reversed(range(start, stop)), position=0):
                t = (torch.ones(num_images) * i).long().to(self.device)
                predicted_noise = model(predicted_chain, t)
                predicted_chain = self.ddpm_mu_t(predicted_chain, predicted_noise, t)
                correct_chain = self.ddpm_mean_t(correct_chain, t, x_0=images)
                if i % skip == 0 or i == 1:
                    d_t = (correct_chain - predicted_chain) ** 2
                    dts[:, int((i - start) / skip)] = d_t
                    t = (torch.ones(num_images) * i - 1).long().to(self.device)
                    x_t, noise = self.noise_images(images, t, pyramid=pyramid, discount=discount)
                    predicted_chain = x_t
                    correct_chain = x_t
        return dts

    def deviations_noise(self, model, images, start=75, stop=200, pyramid=False, discount = 0.8):
        """This method calculates the deviations for each time step t in the interval T_l, T_u on the noise level.
        This provides similar results.

        Parameters
        ----------
        model : _type_
            The learned DDPM model (U-Net)
        images : tensor
            The samples x_0
        start : int, optional
            Lower endpoint of interval T_l, by default 75
        stop : int, optional
            Upper endpoint of interval T_u, by default 200
        pyramid : bool, optional
            flag that decides if pyramid noise is used, by default False
        discount : float, optional
            the discount for the pyramid noise, by default 0.8

        Returns
        -------
        tensor
            The tensor that contains the deviations for each time step in the dimension with the index 1
        """
        if stop is None:
            stop = self.noise_steps
        if start == 0:  # The start can not be the original sample x_0
            start = 1
        num_images = images.shape[0]
        model.eval()
        with torch.no_grad():
            dts = torch.zeros(
                (
                    num_images,
                    stop - start,
                    images.shape[1],
                    images.shape[2],
                    images.shape[3],
                )
            ).to(self.device)

            for i in tqdm(reversed(range(start, stop)), position=0):
                t = (torch.ones(num_images) * i).long().to(self.device)
                x_t, noise = self.noise_images(images, t, pyramid=pyramid, discount=discount)
                predicted_noise = model(x_t, t)
                d_t = (predicted_noise - noise) ** 2
                dts[:, i - start] = d_t
        return dts

