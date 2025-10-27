static inline void put_codeword(PutBitContext *pb, vorbis_enc_codebook *cb,

                                int entry)

{

    assert(entry >= 0);

    assert(entry < cb->nentries);

    assert(cb->lens[entry]);

    put_bits(pb, cb->lens[entry], cb->codewords[entry]);

}
