const uint8_t *get_submv_prob(uint32_t left, uint32_t top)

{

    if (left == top)

        return vp8_submv_prob[4 - !!left];

    if (!top)

        return vp8_submv_prob[2];

    return vp8_submv_prob[1 - !!left];

}
