from paired_xray_mask_loader import build_pairs_data_proposed, PairedXrayMaskSequence

root = r"G:\projects\paper 25\data_proposed"

train_pairs, class_to_idx = build_pairs_data_proposed(root, "train")
val_pairs, _ = build_pairs_data_proposed(root, "validation")
test_pairs, _ = build_pairs_data_proposed(root, "test")

train_seq = PairedXrayMaskSequence(train_pairs, img_size=299, batch_size=4, x_channels=1, shuffle=True)
val_seq   = PairedXrayMaskSequence(val_pairs,   img_size=299, batch_size=4, x_channels=1, shuffle=False)

x, y, m = train_seq[0]
print(x.shape, y.shape, m.shape)
print(class_to_idx)

