static inline direntry_t* create_short_and_long_name(BDRVVVFATState* s,

	unsigned int directory_start, const char* filename, int is_dot)

{

    int i,j,long_index=s->directory.next;

    direntry_t* entry = NULL;

    direntry_t* entry_long = NULL;



    if(is_dot) {

	entry=array_get_next(&(s->directory));

	memset(entry->name,0x20,11);

	memcpy(entry->name,filename,strlen(filename));

	return entry;

    }



    entry_long=create_long_filename(s,filename);



    i = strlen(filename);

    for(j = i - 1; j>0  && filename[j]!='.';j--);

    if (j > 0)

	i = (j > 8 ? 8 : j);

    else if (i > 8)

	i = 8;



    entry=array_get_next(&(s->directory));

    memset(entry->name,0x20,11);

    memcpy(entry->name, filename, i);



    if(j > 0)

	for (i = 0; i < 3 && filename[j+1+i]; i++)

	    entry->extension[i] = filename[j+1+i];



    /* upcase & remove unwanted characters */

    for(i=10;i>=0;i--) {

	if(i==10 || i==7) for(;i>0 && entry->name[i]==' ';i--);

	if(entry->name[i]<=' ' || entry->name[i]>0x7f

		|| strchr(".*?<>|\":/\\[];,+='",entry->name[i]))

	    entry->name[i]='_';

        else if(entry->name[i]>='a' && entry->name[i]<='z')

            entry->name[i]+='A'-'a';

    }



    /* mangle duplicates */

    while(1) {

	direntry_t* entry1=array_get(&(s->directory),directory_start);

	int j;



	for(;entry1<entry;entry1++)

	    if(!is_long_name(entry1) && !memcmp(entry1->name,entry->name,11))

		break; /* found dupe */

	if(entry1==entry) /* no dupe found */

	    break;



	/* use all 8 characters of name */

	if(entry->name[7]==' ') {

	    int j;

	    for(j=6;j>0 && entry->name[j]==' ';j--)

		entry->name[j]='~';

	}



	/* increment number */

	for(j=7;j>0 && entry->name[j]=='9';j--)

	    entry->name[j]='0';

	if(j>0) {

	    if(entry->name[j]<'0' || entry->name[j]>'9')

	        entry->name[j]='0';

	    else

	        entry->name[j]++;

	}

    }



    /* calculate checksum; propagate to long name */

    if(entry_long) {

        uint8_t chksum=fat_chksum(entry);



	/* calculate anew, because realloc could have taken place */

	entry_long=array_get(&(s->directory),long_index);

	while(entry_long<entry && is_long_name(entry_long)) {

	    entry_long->reserved[1]=chksum;

	    entry_long++;

	}

    }



    return entry;

}
