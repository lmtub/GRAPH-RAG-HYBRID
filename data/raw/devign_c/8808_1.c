pp_mode *pp_get_mode_by_name_and_quality(const char *name, int quality)

{

    char temp[GET_MODE_BUFFER_SIZE];

    char *p= temp;

    static const char filterDelimiters[] = ",/";

    static const char optionDelimiters[] = ":|";

    struct PPMode *ppMode;

    char *filterToken;



    if (!name)  {

        av_log(NULL, AV_LOG_ERROR, "pp: Missing argument\n");

        return NULL;




    if (!strcmp(name, "help")) {

        const char *p;

        for (p = pp_help; strchr(p, '\n'); p = strchr(p, '\n') + 1) {

            av_strlcpy(temp, p, FFMIN(sizeof(temp), strchr(p, '\n') - p + 2));

            av_log(NULL, AV_LOG_INFO, "%s", temp);


        return NULL;




    ppMode= av_malloc(sizeof(PPMode));



    ppMode->lumMode= 0;

    ppMode->chromMode= 0;

    ppMode->maxTmpNoise[0]= 700;

    ppMode->maxTmpNoise[1]= 1500;

    ppMode->maxTmpNoise[2]= 3000;

    ppMode->maxAllowedY= 234;

    ppMode->minAllowedY= 16;

    ppMode->baseDcDiff= 256/8;

    ppMode->flatnessThreshold= 56-16-1;

    ppMode->maxClippedThreshold= 0.01;

    ppMode->error=0;



    memset(temp, 0, GET_MODE_BUFFER_SIZE);

    av_strlcpy(temp, name, GET_MODE_BUFFER_SIZE - 1);



    av_log(NULL, AV_LOG_DEBUG, "pp: %s\n", name);



    for(;;){

        char *filterName;

        int q= 1000000; //PP_QUALITY_MAX;

        int chrom=-1;

        int luma=-1;

        char *option;

        char *options[OPTIONS_ARRAY_SIZE];

        int i;

        int filterNameOk=0;

        int numOfUnknownOptions=0;

        int enable=1; //does the user want us to enabled or disabled the filter



        filterToken= strtok(p, filterDelimiters);

        if(filterToken == NULL) break;

        p+= strlen(filterToken) + 1; // p points to next filterToken

        filterName= strtok(filterToken, optionDelimiters);





        av_log(NULL, AV_LOG_DEBUG, "pp: %s::%s\n", filterToken, filterName);



        if(*filterName == '-'){

            enable=0;

            filterName++;




        for(;;){ //for all options

            option= strtok(NULL, optionDelimiters);

            if(option == NULL) break;



            av_log(NULL, AV_LOG_DEBUG, "pp: option: %s\n", option);

            if(!strcmp("autoq", option) || !strcmp("a", option)) q= quality;

            else if(!strcmp("nochrom", option) || !strcmp("y", option)) chrom=0;

            else if(!strcmp("chrom", option) || !strcmp("c", option)) chrom=1;

            else if(!strcmp("noluma", option) || !strcmp("n", option)) luma=0;

            else{

                options[numOfUnknownOptions] = option;

                numOfUnknownOptions++;


            if(numOfUnknownOptions >= OPTIONS_ARRAY_SIZE-1) break;


        options[numOfUnknownOptions] = NULL;



        /* replace stuff from the replace Table */

        for(i=0; replaceTable[2*i]!=NULL; i++){

            if(!strcmp(replaceTable[2*i], filterName)){

                int newlen= strlen(replaceTable[2*i + 1]);

                int plen;

                int spaceLeft;



                p--, *p=',';



                plen= strlen(p);

                spaceLeft= p - temp + plen;

                if(spaceLeft + newlen  >= GET_MODE_BUFFER_SIZE - 1){




                memmove(p + newlen, p, plen+1);

                memcpy(p, replaceTable[2*i + 1], newlen);

                filterNameOk=1;





        for(i=0; filters[i].shortName!=NULL; i++){

            if(   !strcmp(filters[i].longName, filterName)

               || !strcmp(filters[i].shortName, filterName)){

                ppMode->lumMode &= ~filters[i].mask;

                ppMode->chromMode &= ~filters[i].mask;



                filterNameOk=1;

                if(!enable) break; // user wants to disable it



                if(q >= filters[i].minLumQuality && luma)

                    ppMode->lumMode|= filters[i].mask;

                if(chrom==1 || (chrom==-1 && filters[i].chromDefault))

                    if(q >= filters[i].minChromQuality)

                            ppMode->chromMode|= filters[i].mask;



                if(filters[i].mask == LEVEL_FIX){

                    int o;

                    ppMode->minAllowedY= 16;

                    ppMode->maxAllowedY= 234;

                    for(o=0; options[o]!=NULL; o++){

                        if(  !strcmp(options[o],"fullyrange")

                           ||!strcmp(options[o],"f")){

                            ppMode->minAllowedY= 0;

                            ppMode->maxAllowedY= 255;

                            numOfUnknownOptions--;




                else if(filters[i].mask == TEMP_NOISE_FILTER)

                {

                    int o;

                    int numOfNoises=0;



                    for(o=0; options[o]!=NULL; o++){

                        char *tail;

                        ppMode->maxTmpNoise[numOfNoises]=

                            strtol(options[o], &tail, 0);

                        if(tail!=options[o]){

                            numOfNoises++;

                            numOfUnknownOptions--;

                            if(numOfNoises >= 3) break;




                else if(filters[i].mask == V_DEBLOCK   || filters[i].mask == H_DEBLOCK

                     || filters[i].mask == V_A_DEBLOCK || filters[i].mask == H_A_DEBLOCK){

                    int o;



                    for(o=0; options[o]!=NULL && o<2; o++){

                        char *tail;

                        int val= strtol(options[o], &tail, 0);

                        if(tail==options[o]) break;



                        numOfUnknownOptions--;

                        if(o==0) ppMode->baseDcDiff= val;

                        else ppMode->flatnessThreshold= val;



                else if(filters[i].mask == FORCE_QUANT){

                    int o;

                    ppMode->forcedQuant= 15;



                    for(o=0; options[o]!=NULL && o<1; o++){

                        char *tail;

                        int val= strtol(options[o], &tail, 0);

                        if(tail==options[o]) break;



                        numOfUnknownOptions--;

                        ppMode->forcedQuant= val;





        if(!filterNameOk) ppMode->error++;

        ppMode->error += numOfUnknownOptions;




    av_log(NULL, AV_LOG_DEBUG, "pp: lumMode=%X, chromMode=%X\n", ppMode->lumMode, ppMode->chromMode);

    if(ppMode->error){

        av_log(NULL, AV_LOG_ERROR, "%d errors in postprocess string \"%s\"\n", ppMode->error, name);

        av_free(ppMode);

        return NULL;


    return ppMode;
