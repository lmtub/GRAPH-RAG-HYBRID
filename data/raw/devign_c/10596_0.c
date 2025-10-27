void av_aes_crypt(AVAES *a, uint8_t *dst_, const uint8_t *src_,

                  int count, uint8_t *iv_, int decrypt)

{

    av_aes_block       *dst = (av_aes_block *) dst_;

    const av_aes_block *src = (const av_aes_block *) src_;

    av_aes_block        *iv = (av_aes_block *) iv_;



    while (count--) {

        addkey(&a->state[1], src, &a->round_key[a->rounds]);

        if (decrypt) {

            crypt(a, 0, inv_sbox, dec_multbl);

            if (iv) {

                addkey(&a->state[0], &a->state[0], iv);

                memcpy(iv, src, 16);

            }

            addkey(dst, &a->state[0], &a->round_key[0]);

        } else {

            if (iv)

                addkey(&a->state[1], &a->state[1], iv);

            crypt(a, 2, sbox, enc_multbl);

            addkey(dst, &a->state[0], &a->round_key[0]);

            if (iv)

                memcpy(iv, dst, 16);

        }

        src++;

        dst++;

    }

}
