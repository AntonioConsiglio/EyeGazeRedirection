'''

implementation in pytorch of paper "Photo-Realistic Monocular Gaze Redirection Using Generative Adversarial Networks" 
Link to the paper for more info: https://arxiv.org/abs/1903.12530

'''
import os
import torch
from torch import nn
import torch.nn.functional as F
from model.layers import ConvBNReLU,ResBlock, DeConvBNReLU,ConvLReLU
import torchvision
from tqdm import tqdm
from torch.utils.tensorboard.writer import SummaryWriter

class CustomVGG(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg16 = torchvision.models.vgg16(torchvision.models.VGG16_Weights.DEFAULT)
        del self.vgg16.classifier
        self.conv1 = self.vgg16.features[:2]
        self.mpool1 = self.vgg16.features[2]
        self.conv2 = self.vgg16.features[3:5]
        self.mpool2 = self.vgg16.features[5]
        self.conv3 = self.vgg16.features[6:9]
        self.mpool3 = self.vgg16.features[9]
        self.conv4 = self.vgg16.features[10:13]
        self.mpool4 = self.vgg16.features[13]
        self.conv5 = self.vgg16.features[14:17]

    def forward(self,x):
        conv12 = self.conv1(x)
        conv22 = self.conv2(self.mpool1(conv12))
        conv33 = self.conv3(self.mpool2(conv22))
        conv43 = self.conv4(self.mpool3(conv33))
        conv53 = self.conv5(self.mpool4(conv43))

        return {"conv12":conv12,"conv22":conv22,"conv33":conv33,"conv43":conv43,"conv53":conv53}


class GazeCorrection(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.generator = Generator()
        self.discriminator = Discriminator()
        self.vgg16 = CustomVGG()

    def forward(self, x, angles):
        """Forward pass through the model.

        Parameters
        ----------
        x: torch.Tensor
            Input tensor.
        angles: torch.Tensor
            Gaze angles tensor.

        Returns
        -------
        x_g: torch.Tensor
            Output tensor from the generator.
        """
        # Concatenate input tensor and gaze angles

        # Pass concatenated input through the generator
        x_g = self.generator(x,angles)
        
        return x_g
    
    def train_(self,train_loader,optimizers,schedulers,restore_chkpt=None,
               summary_writer:SummaryWriter=None,logspath=None,epochs=200):
        """Train the model and save checkpoints."""
        start_epoch = 1
        self.logpath = logspath
        self.summary_writer = summary_writer
        self.last_name = "last_weights_1.pth"
        self.dataset = train_loader.dataset

        #get loss obj
        self.nloss1 = nn.L1Loss()

        self.nmloss1 = nn.MSELoss()
        self.nmloss2 = nn.MSELoss()

        self.reglossmls = nn.MSELoss()
        self.deglossmls = nn.MSELoss()

        # to device
        self.to(self.device)
        # Initialize summary writer and checkpoint saver
        if restore_chkpt is not None:
            chk = torch.load(restore_chkpt,map_location=self.device)
            optimizers["d"].load_state_dict(chk["d_optim"])
            optimizers["g"].load_state_dict(chk["g_optim"])
            if schedulers is not None:
                schedulers["d"].load_state_dict(chk["d_l_scheduler"])
                schedulers["g"].load_state_dict(chk["g_l_scheduler"])
            self.load_state_dict(chk["model_state_dict"])
            start_epoch = chk["last_epoch"]+1
            self.last_name = f"last_weights_{chk['last_epoch']}.pth"
        # summary_writer = torch.summary.FileWriter(os.path.join(self.params['log_dir'], 'summary'))
        
        # Set model to training mode
        self.train()

        try:
            for epoch in range(start_epoch,epochs+1,1):
                average_g_loss = 0
                average_d_loss = 0
                average_total_d_loss = 0
                average_total_g_loss = 0
                average_recon_loss = 0
                averate_style_loss = 0
                average_content_loss = 0
                average_reg_loss_g = 0

                loop = tqdm(train_loader,bar_format='{l_bar}{bar:20}{r_bar}{bar:-10b}')
                loop.set_description(f"epoch = {epoch}/{epochs}")
                # if epoch >= num_epoch / 2:
                #     learning_rate = (2. - 2. * epoch / num_epoch) * self.params['lr']
                # Zero the gradients
                for key,opt in optimizers.items():
                    opt.zero_grad()
                
                total_it = 1 + (len(loop) // 5) * 5 

                for it, (real_images,r_angles,_,target_images,t_angles) in enumerate(loop,start=1):

                    if it == total_it+1:
                        break
                    # Convert data to PyTorch tensors
                    real_images = real_images.to(self.device)
                    r_angles = r_angles.to(self.device)
                    target_images = target_images.to(self.device)
                    t_angles = t_angles.to(self.device)

                    # Zero the gradients
                    for key,opt in optimizers.items():
                        if key == "d":
                            opt.zero_grad()                           

                    # Forward pass
                    x_g = self.forward(real_images,t_angles)
                    x_recon = self.forward(x_g,r_angles)

                    # Compute losses
                    recon_loss = self.nloss1(x_recon, real_images)
                    c_loss, s_loss = self.feat_loss(x_g, target_images)
                    d_loss, g_loss, reg_loss_d, reg_loss_g, gp = self.adv_loss(real_images, x_g,r_angles,t_angles)

                    # Overall loss
                    total_g_loss = g_loss + 5.0 * reg_loss_g + 50.0 * recon_loss + 100.0 * s_loss + 100.0 * c_loss
                    total_d_loss = d_loss + 5.0 * reg_loss_d

                    # summ overall loss
                    average_total_g_loss += total_g_loss.item()
                    average_total_d_loss += total_d_loss.item()
                    average_d_loss += d_loss.item()
                    average_g_loss += g_loss.item()
                    average_recon_loss += recon_loss.item()
                    averate_style_loss += s_loss.item()
                    average_content_loss += c_loss.item()
                    average_reg_loss_g += reg_loss_g.item()

                    # Backward pass
                    total_g_loss.backward(retain_graph=True)
                    total_d_loss.backward()

                    # Update weights
                    for key,opt in optimizers.items():
                        if key == "d":
                            opt.step()
                        elif key == "g" and it % 5 == 0:
                            opt.step()
                            opt.zero_grad()
                    
                    if it == total_it//2:
                        self._write_example_result(summary_writer,x_g.detach(),x_recon.detach(),real_images.detach(),
                                           target_images.detach(),epoch=epoch,example_name="Middle")

                    loop.set_postfix(total_d_loss=total_d_loss.item(),total_g_loss=total_g_loss.item())

                average_g_loss /= it
                average_d_loss /= it
                average_total_d_loss /= it
                average_total_g_loss /= it
                average_recon_loss /= it
                averate_style_loss /= it
                average_content_loss /= it
                average_reg_loss_g /= it

                summary_writer.add_scalar("LOSS/_d_loss",average_d_loss,epoch)
                summary_writer.add_scalar("LOSS/_g_loss",average_g_loss,epoch)
                summary_writer.add_scalar("LOSS/_regression_loss",average_reg_loss_g,epoch)
                summary_writer.add_scalar("LOSS/_reconstruction_loss",average_recon_loss,epoch)
                summary_writer.add_scalar("LOSS/_style_loss",averate_style_loss,epoch)
                summary_writer.add_scalar("LOSS/_content_loss",average_content_loss,epoch)
                

                summary_writer.add_scalar("TOTAL_LOSS/total_d_loss",average_total_d_loss,epoch)
                summary_writer.add_scalar("TOTAL_LOSS/total_g_loss",average_total_g_loss,epoch)
                summary_writer.add_scalar("LR",schedulers["d"].get_last_lr()[0],epoch)


                if epoch % 5 == 0 or epoch == start_epoch:
                    self._save_checkpoint(optimizers,schedulers,epoch)

                
                self._write_example_result(summary_writer,x_g.detach(),x_recon.detach(),real_images.detach(),
                                           target_images.detach(),epoch=epoch,example_name="Last")

                
                if epoch == start_epoch:
                    torch.save(self.state_dict(),os.path.join(logspath,"weights",self.last_name))
                else:
                    os.remove(os.path.join(logspath,"weights",self.last_name))
                    self.last_name = self.last_name.replace(f"{epoch-1}",f"{epoch}")
                    torch.save(self.state_dict(),os.path.join(logspath,"weights",self.last_name))

                for key,scheduler in schedulers.items():
                    scheduler.step()
                    
        except KeyboardInterrupt:
            print("Training stopped")
    
    def _write_example_result(self,summary_writer:SummaryWriter,x_g,x_recon,real_images,target_images,epoch,example_name):
        x_g = self.dataset.denormalize_color(x_g.cpu())
        x_recon = self.dataset.denormalize_color(x_recon.cpu())
        real_images = self.dataset.denormalize_color(real_images.cpu())
        target_images = self.dataset.denormalize_color(target_images.cpu())
        black = torch.zeros_like(x_g)[:,:,:,:32]
        grid2show = torch.cat([x_g,black,target_images,black,x_recon,black,real_images],dim=-1)

        grid = torchvision.utils.make_grid(grid2show,1,padding=5,pad_value=1.0)
        summary_writer.add_image(f"Image Examples {example_name}",grid,epoch)

    def _save_checkpoint(self,optimizers,schedulers=None,epoch=0):
        
        checkpoint = {
            "model_state_dict":self.state_dict(),
            "d_optim":optimizers["d"].state_dict(),
            "g_optim":optimizers["g"].state_dict(),
            "d_l_scheduler": schedulers["d"].state_dict() if schedulers is not None else None,
            "g_l_scheduler": schedulers["g"].state_dict() if schedulers is not None else None,
            "last_epoch":epoch,
        }
        
        checkpoint_name = f"chekpoints{epoch}.pt"
        check_path = os.path.join(self.logpath,"checkpoints",checkpoint_name)
        torch.save(checkpoint,check_path)
        
        
    
    def feat_loss(self, generated_images, target_images):
        """Compute feature loss.

        Parameters
        ----------
        generated_images: torch.Tensor
            Batch of generated images.
        target_images: torch.Tensor
            Batch of target images.

        Returns
        -------
        c_loss: torch.Tensor
            Content loss.
        s_loss: torch.Tensor
            Style loss.
        """
        # Forward pass through VGG16 model for both generated and target images
        gen_feats = self.vgg16(generated_images)
        target_feats = self.vgg16(target_images)

        # Extract features for content and style layers
        content_layers = ["conv53"]
        style_layers = ["conv12", "conv22", "conv33", "conv43"]

        # Compute content loss
        feat_a = gen_feats[content_layers[0]]
        feat_b = target_feats[content_layers[0]]
        size = torch.numel(feat_a)
        c_loss =  F.mse_loss(feat_a,feat_b ) * 2 / size

        # Compute style loss
        s_loss = 0.0
        for layer in style_layers:
            size = torch.numel(gen_feats[layer])
            gram_a = self.gram_matrix(gen_feats[layer])
            gram_b = self.gram_matrix(target_feats[layer])
            s_loss += F.mse_loss(gram_a, gram_b) * 2 / size

        return c_loss, s_loss
    
    # Assuming you have a gram_matrix function defined:
    def gram_matrix(self,input_tensor):
        batch_size, channels, height, width = input_tensor.size()
        features = input_tensor.view(batch_size * channels, height * width)
        gram_matrix = torch.mm(features, features.t())
        return gram_matrix.div(batch_size * channels * height * width)

    def adv_loss(self, real_images, fake_images,start_angle,target_angle):
        """Compute adversarial loss.

        Parameters
        ----------
        real_images: torch.Tensor
            Batch of real images.
        fake_images: torch.Tensor
            Batch of generated images.

        Returns
        -------
        d_loss: torch.Tensor
            Adversarial loss for training the discriminator.
        g_loss: torch.Tensor
            Adversarial loss for training the generator.
        reg_loss_d: torch.Tensor
            Mean squared error loss for training gaze estimator (discriminator).
        reg_loss_g: torch.Tensor
            Mean squared error loss for training gaze estimator (generator).
        gp: torch.Tensor
            Gradient penalty.
        """

        # Forward pass through the discriminator for real and fake images
        gan_real, reg_real = self.discriminator(real_images)
        gan_fake, reg_fake = self.discriminator(fake_images)  # Detach to avoid backpropagating through the generator

        # Compute gradient penalty
        eps = torch.rand(real_images.size(0), 1, 1, 1,device = self.device)
        interpolated = eps * real_images + (1. - eps) * fake_images
      
        gan_inter, _ = self.discriminator(interpolated)
        grad = torch.autograd.grad(outputs=gan_inter, inputs=interpolated, grad_outputs=torch.ones(gan_inter.size(),device = self.device),
                                   create_graph=True, retain_graph=True)[0]
        
        slopes = torch.sqrt(torch.sum(grad**2, dim=[1, 2, 3]))
        gp = torch.mean((slopes - 1.)**2) # gradient penalty

        # Compute adversarial and regression losses
        d_loss = (-torch.mean(gan_real) + torch.mean(gan_fake) + 10. * gp)
        g_loss = -torch.mean(gan_fake)
        reg_loss_d = self.deglossmls(reg_real, start_angle) #MSELoss
        reg_loss_g = self.reglossmls(reg_fake, target_angle) #MSELoss

        return d_loss, g_loss, reg_loss_d, reg_loss_g, gp

    

class Generator(nn.Module):
    def __init__(self,in_channels=5,weights=None):
        super().__init__()
        self.layer_channel = [64,128,*[256]*7,128,64,3]

        self.layers = nn.ModuleList()
        for n,ch in enumerate(self.layer_channel):
            if n == 0:
                self.layers.append(ConvBNReLU(in_channels=in_channels,
                                                out_channels=ch,
                                                 kernel_size=(7,7),
                                                stride=1,
                                                padding=3))
            elif  0 < n < 3:
                self.layers.append(ConvBNReLU(in_channels=in_channels,
                                            out_channels=ch,
                                            kernel_size=(4,4),
                                            stride=2,
                                            padding=1))
            elif n < 9:
                self.layers.append(ResBlock(in_channels=in_channels,
                                            out_channels=ch,
                                            kernel_size=(3,3),
                                            stride=1,
                                            padding=1))
            elif n < 11:
                self.layers.append(DeConvBNReLU(in_channels=in_channels,
                                            out_channels=ch,
                                            kernel_size=(4,4),
                                            stride=2,
                                            padding=1))
            else:
                self.layers.append(nn.Conv2d(in_channels=in_channels,
                                            out_channels=ch,
                                            kernel_size=(7,7),
                                            stride=1,
                                            padding=3))

            in_channels = ch

        self._init_weights(weights)

    def forward(self,x,angles):

        angles_reshaped = angles.view(-1, angles.size(-1), 1, 1)
        angles_tiled = angles_reshaped.repeat(1,1,x.size(2), x.size(3))
        x = torch.cat([x, angles_tiled], dim=1)

        for l in self.layers:
            x = l(x)
    
        out = torch.tanh(x)

        return out

    def _init_weights(self,pre_weights):
        if pre_weights is not None:
            weights = torch.load(pre_weights,map_location="cpu")
            self.load_state_dict(weights)
        else:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

class Discriminator(nn.Module):
    def __init__(self,in_channels=3,n_layers=5):
        super().__init__()
        base_out = 64
        self.layers_channel = n_layers
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(ConvLReLU(in_channels=in_channels,
                                         out_channels=base_out,
                                         kernel_size=(4,4),stride=2,padding=1,
                                         bias=True))
            in_channels = base_out
            base_out *=2
        
        # Additional layers for gaze estimation
        self.logit_gan = nn.Conv2d(base_out//2, 1, kernel_size=2, stride=1, padding=1, bias=False) #discriminator architecture
        self.logit_reg = nn.Conv2d(base_out//2, 2, kernel_size=2, stride=1, padding=0, bias=False) #gaze estimator

        self._init_weights(pre_weights=None)
    
    def forward(self,x):

        for layer in self.layers:
            x = layer(x)
        
        x_gan = self.logit_gan(x)
        x_reg = self.logit_reg(x).view(-1, 2)

        return x_gan, x_reg
    
    def _init_weights(self,pre_weights):
        if pre_weights is not None:
            weights = torch.load(pre_weights,map_location="cpu")
            self.load_state_dict(weights)
        else:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')



if __name__ == "__main__":
    generator = GazeCorrection()

    print(generator)
    x = torch.rand((4,3,64,64))
    angles = torch.tensor([[5,5]])
    angles = angles.repeat((4,1))

    generator(x,angles)
    pass