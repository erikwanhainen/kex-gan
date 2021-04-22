import os
for i in range(1, 63, 2):
    command1 = f'scp -r -v -o "IdentitiesOnly=yes" -i ~/.ssh/id_rsa_personal -P 56027 da150x_14@130.237.37.22:~/kex-gan/src/checkpoints/w_128/ckpt-{i}.data-00000-of-00001 src/checkpoints/w_128_v2/w_128'
    command2 = f'scp -r -v -o "IdentitiesOnly=yes" -i ~/.ssh/id_rsa_personal -P 56027 da150x_14@130.237.37.22:~/kex-gan/src/checkpoints/w_128/ckpt-{i}.index src/checkpoints/w_128_v2/w_128'
    print('#################################')
    print(i)
    os.system(command1)
    os.system(command2)
