static inline uint8_t fat_chksum(const direntry_t* entry)

{

    uint8_t chksum=0;

    int i;



    for(i=0;i<11;i++) {

        unsigned char c;



        c = (i < 8) ? entry->name[i] : entry->extension[i-8];

        chksum=(((chksum&0xfe)>>1)|((chksum&0x01)?0x80:0)) + c;

    }



    return chksum;

}
