import torch

from tqdm import tqdm
import matplotlib.pyplot as plt

def train(generator, discriminator):
	# TRAINING
	generator.train()
	discriminator.train()
	training_progress = tqdm(total = dataset.train_cap * args.epochs, desc = "Training")

	try:
		for epoch in range(args.epochs):
			for real_sample in training_iterator:
				print(dataset.decode(real_sample))
				assert False
				fake_sample = generator.generate(args.batch_size)

				# update the progress bar
				training_progress.update(args.batch_size)

				# take outputs from discriminator
				score_real = discriminator(real_sample)
				score_fake = discriminator(fake_sample)

				# Save scores for evaluation
				discriminator.scores_real.append(score_real.item())
				discriminator.scores_fake.append(score_fake.item())

				# calculate loss
				loss_d = torch.mean(-torch.log(1 - score_fake) - torch.log(score_real))
				discriminator.losses.append(loss_d.item())

				# optimize discriminator
				discriminator.learn(loss_d)
				generator.learn(fake_sample, discriminator)

	finally:
		# Models are always saved, even after a KeyboardInterrupt
		generator.saveModel()
		discriminator.saveModel()

		# # TESTING
		# discriminator.eval()
		# generator.eval()

		# with torch.no_grad():
		# 	haikus = dataset.decode(generator.generate(10))
		# 	for haiku in haikus:
		# 		print(haiku)

		# # smooth out the loss functions (avg of last 25 episodes)
		# generator.losses = [np.mean(generator.losses[max(0, t-25):(t+1)]) for t in range(len(generator.losses))]
		# discriminator.losses = [np.mean(discriminator.losses[max(0, t-25):(t+1)]) for t in range(len(discriminator.losses))]

		# # plot the graph of the different losses over time
		# fig, axs = plt.subplots(2, 2, num = "Training Data")

		# # Discriminator scores
		# axs[0, 0].title.set_text("Discriminator Scores")
		# axs[0, 0].plot(discriminator.scores_real, label = "Real")
		# axs[0, 0].plot(discriminator.scores_fake, label = "Fake")
		# axs[0, 0].legend()

		# # Generator Loss
		# axs[0, 1].title.set_text("Generator Loss")
		# axs[0, 1].plot(generator.losses, label = "Generator Loss")

		# # Discriminator Loss
		# axs[1, 1].title.set_text("Discriminator Loss")
		# axs[1, 1].plot(discriminator.losses, label = "Discriminator Loss")
		# fig.tight_layout()
		# plt.savefig("training_graphs/main")
		# plt.show()

	return generator, discriminator
