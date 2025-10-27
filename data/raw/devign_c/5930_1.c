static int check_for_block_signature(BlockDriverState *bs, const uint8_t *buf)

{

    static const uint8_t signatures[][4] = {

        { 'Q', 'F', 'I', 0xfb }, /* qcow/qcow2 */

        { 'C', 'O', 'W', 'D' }, /* VMDK3 */

        { 'V', 'M', 'D', 'K' }, /* VMDK4 */

        { 'O', 'O', 'O', 'M' }, /* UML COW */

        {}

    };

    int i;



    for (i = 0; signatures[i][0] != 0; i++) {

        if (memcmp(buf, signatures[i], 4) == 0) {

            return 1;

        }

    }



    return 0;

}
