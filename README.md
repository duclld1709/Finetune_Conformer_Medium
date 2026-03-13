1. Replace d_model by d_head (Done)
2. Log total model's parameters (Done)
3. Computer global mean and std
4. Batch_size 16, Warmup Step 12k (Setting while running)
5. Debug CTCLoss (Done)
6. Beam_size 10 chỉ khi evaluate 5 sample predictions
7. Noam Scheduler Checking (Done)
8. Gradient Clipping (Done)
9. Adjust appropriate SpecAug for get_transform method (Done)
