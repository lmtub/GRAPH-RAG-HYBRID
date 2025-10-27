type_init(boston_register_types)



static void gen_firmware(uint32_t *p, hwaddr kernel_entry, hwaddr fdt_addr,

                         bool is_64b)

{

    const uint32_t cm_base = 0x16100000;

    const uint32_t gic_base = 0x16120000;

    const uint32_t cpc_base = 0x16200000;



    /* Move CM GCRs */

    if (is_64b) {

        stl_p(p++, 0x40287803);                 /* dmfc0 $8, CMGCRBase */

        stl_p(p++, 0x00084138);                 /* dsll $8, $8, 4 */

    } else {

        stl_p(p++, 0x40087803);                 /* mfc0 $8, CMGCRBase */

        stl_p(p++, 0x00084100);                 /* sll  $8, $8, 4 */

    }

    stl_p(p++, 0x3c09a000);                     /* lui  $9, 0xa000 */

    stl_p(p++, 0x01094025);                     /* or   $8, $9 */

    stl_p(p++, 0x3c0a0000 | (cm_base >> 16));   /* lui  $10, cm_base >> 16 */

    if (is_64b) {

        stl_p(p++, 0xfd0a0008);                 /* sd   $10, 0x8($8) */

    } else {

        stl_p(p++, 0xad0a0008);                 /* sw   $10, 0x8($8) */

    }

    stl_p(p++, 0x012a4025);                     /* or   $8, $10 */



    /* Move & enable GIC GCRs */

    stl_p(p++, 0x3c090000 | (gic_base >> 16));  /* lui  $9, gic_base >> 16 */

    stl_p(p++, 0x35290001);                     /* ori  $9, 0x1 */

    if (is_64b) {

        stl_p(p++, 0xfd090080);                 /* sd   $9, 0x80($8) */

    } else {

        stl_p(p++, 0xad090080);                 /* sw   $9, 0x80($8) */

    }



    /* Move & enable CPC GCRs */

    stl_p(p++, 0x3c090000 | (cpc_base >> 16));  /* lui  $9, cpc_base >> 16 */

    stl_p(p++, 0x35290001);                     /* ori  $9, 0x1 */

    if (is_64b) {

        stl_p(p++, 0xfd090088);                 /* sd   $9, 0x88($8) */

    } else {

        stl_p(p++, 0xad090088);                 /* sw   $9, 0x88($8) */

    }



    /*

     * Setup argument registers to follow the UHI boot protocol:

     *

     * a0/$4 = -2

     * a1/$5 = virtual address of FDT

     * a2/$6 = 0

     * a3/$7 = 0

     */

    stl_p(p++, 0x2404fffe);                     /* li   $4, -2 */

                                                /* lui  $5, hi(fdt_addr) */

    stl_p(p++, 0x3c050000 | ((fdt_addr >> 16) & 0xffff));

    if (fdt_addr & 0xffff) {                    /* ori  $5, lo(fdt_addr) */

        stl_p(p++, 0x34a50000 | (fdt_addr & 0xffff));

    }

    stl_p(p++, 0x34060000);                     /* li   $6, 0 */

    stl_p(p++, 0x34070000);                     /* li   $7, 0 */



    /* Load kernel entry address & jump to it */

                                                /* lui  $25, hi(kernel_entry) */

    stl_p(p++, 0x3c190000 | ((kernel_entry >> 16) & 0xffff));

                                                /* ori  $25, lo(kernel_entry) */

    stl_p(p++, 0x37390000 | (kernel_entry & 0xffff));

    stl_p(p++, 0x03200009);                     /* jr   $25 */

}
