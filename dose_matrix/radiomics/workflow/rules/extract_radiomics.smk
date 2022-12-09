
## Extract radiomics
rule images_nii:
    input:
        get_newdosi_files
    output:
        NII_DIR + "{newdosi_patient}_ID2013A.nii.gz",
        NII_DIR + "{newdosi_patient}_mask_t.nii.gz",
        NII_DIR + "{newdosi_patient}_mask_super_t.nii.gz"
    run:
        logger = setup_logger("csv2nii", NII_DIR + "csv2nii.log", mode_file = "a", creation_msg = False)
        logger.info(f"{wildcards.newdosi_patient}: creation of nii images")
        subdir = os.path.dirname(wildcards.newdosi_patient)
        list_newdosi_files_patient = get_newdosi_files(wildcards)
        list_filenames = [os.path.basename(newdosi_file) for newdosi_file in list_newdosi_files_patient]
        path_csv = DOSES_DATASET_DIR + subdir + "/"
        path_nii = NII_DIR + subdir + "/"
        csv2nii.to_nii(path_csv, path_nii, list_filenames)

rule write_header_radiomics:
    output:
        RADIOMICS_DIR + "header.csv",
        RADIOMICS_DIR + "nbr_features_per_label"
    run:
        feature_extractor.write_header(LABELS_SUPER_T_VOI, LABELS_T_VOI, RADIOMICS_DIR, RADIOMICS_PARAMS_FILE)

rule compute_radiomics:
    input:
        RADIOMICS_DIR + "header.csv",
        RADIOMICS_DIR + "nbr_features_per_label",
        NII_DIR + "{newdosi_patient}_ID2013A.nii.gz",
        NII_DIR + "{newdosi_patient}_mask_t.nii.gz",
        NII_DIR + "{newdosi_patient}_mask_super_t.nii.gz"
    output:
        RADIOMICS_DIR + "{newdosi_patient}_radiomics.csv"
    run:
        logger = setup_logger("feature_extractor", RADIOMICS_DIR + "feature_extraction.log",
                              mode_file = "a", creation_msg = False)
        logger.info(f"{wildcards.newdosi_patient}: Extraction of radiomics")
        newdosi_filename = os.path.basename(wildcards.newdosi_patient)
        subdir = os.path.dirname(wildcards.newdosi_patient)
        image_path = NII_DIR + subdir + "/" + newdosi_filename + "_ID2013A.nii.gz"
        mask_t_path = NII_DIR + subdir + "/" + newdosi_filename + "_mask_t.nii.gz"
        mask_super_t_path = NII_DIR + subdir + "/" + newdosi_filename + "_mask_super_t.nii.gz"
        with open(RADIOMICS_DIR + "nbr_features_per_label", 'r') as file_nbr:
            nbr_features_per_label = int(file_nbr.read())
        feature_extractor.compute_radiomics(image_path, mask_super_t_path, mask_t_path,
                                            LABELS_SUPER_T_VOI, LABELS_T_VOI, newdosi_filename, RADIOMICS_DIR, subdir,
                                            RADIOMICS_PARAMS_FILE, nbr_features_per_label)

rule gather_radiomics:
    input:
        expand(RADIOMICS_DIR + "{newdosi_patient}_radiomics.csv", newdosi_patient = list_newdosi_patients)
    output:
        RADIOMICS_DIR + "dose_matrix_radiomics.csv.gz"
    shell:
        cmd_concatenate_radiomics

