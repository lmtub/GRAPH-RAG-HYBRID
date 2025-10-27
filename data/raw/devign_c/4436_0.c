static int smacker_decode_tree(GetBitContext *gb, HuffContext *hc, uint32_t prefix, int length)

{

    if(length > 32) {

        av_log(NULL, AV_LOG_ERROR, "length too long\n");

        return AVERROR_INVALIDDATA;

    }

    if(!get_bits1(gb)){ //Leaf

        if(hc->current >= 256){

            av_log(NULL, AV_LOG_ERROR, "Tree size exceeded!\n");

            return AVERROR_INVALIDDATA;

        }

        if(length){

            hc->bits[hc->current] = prefix;

            hc->lengths[hc->current] = length;

        } else {

            hc->bits[hc->current] = 0;

            hc->lengths[hc->current] = 0;

        }

        hc->values[hc->current] = get_bits(gb, 8);

        hc->current++;

        if(hc->maxlength < length)

            hc->maxlength = length;

        return 0;

    } else { //Node

        int r;

        length++;

        r = smacker_decode_tree(gb, hc, prefix, length);

        if(r)

            return r;

        return smacker_decode_tree(gb, hc, prefix | (1 << (length - 1)), length);

    }

}
