



condition = torch.tensor([365,25,2,2]).to(device)

gen = generate_sequence(decoder_model, start_token=SOS_token, max_length=100, condition_vector=condition, device=device, temperature=0.5)[:-1]
decoded_string = sequence_to_string(gen, input_lang.index2char)
print("Decoded sequence:", decoded_string)