static uint64_t strongarm_gpio_read(void *opaque, hwaddr offset,

                                    unsigned size)

{

    StrongARMGPIOInfo *s = opaque;



    switch (offset) {

    case GPDR:        /* GPIO Pin-Direction registers */

        return s->dir;



    case GPSR:        /* GPIO Pin-Output Set registers */

        DPRINTF("%s: Read from a write-only register 0x" TARGET_FMT_plx "\n",

                        __func__, offset);

        return s->gpsr;    /* Return last written value.  */



    case GPCR:        /* GPIO Pin-Output Clear registers */

        DPRINTF("%s: Read from a write-only register 0x" TARGET_FMT_plx "\n",

                        __func__, offset);

        return 31337;        /* Specified as unpredictable in the docs.  */



    case GRER:        /* GPIO Rising-Edge Detect Enable registers */

        return s->rising;



    case GFER:        /* GPIO Falling-Edge Detect Enable registers */

        return s->falling;



    case GAFR:        /* GPIO Alternate Function registers */

        return s->gafr;



    case GPLR:        /* GPIO Pin-Level registers */

        return (s->olevel & s->dir) |

               (s->ilevel & ~s->dir);



    case GEDR:        /* GPIO Edge Detect Status registers */

        return s->status;



    default:

        printf("%s: Bad offset 0x" TARGET_FMT_plx "\n", __func__, offset);

    }



    return 0;

}
