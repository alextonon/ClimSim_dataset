    def output_weighting(self, output, data_split, just_weights = False):
        '''
        This function does four transformations, and assumes we are using V1 variables:
        [0] Undos the output scaling
        [1] Weight vertical levels by dp/g
        [2] Weight horizontal area of each grid cell by a[x]/mean(a[x])
        [3] Unit conversion to a common energy unit
        '''
        assert data_split in ['train', 'val', 'scoring', 'test'], 'Provided data_split is not valid. Available options are train, val, scoring, and test.'
        num_samples = output.shape[0]
        if just_weights:
            weightings = np.ones(output.shape)

        if not self.full_vars:
            ptend_t = output[:,:60].reshape((int(num_samples/self.num_latlon), self.num_latlon, 60))
            ptend_q0001 = output[:,60:120].reshape((int(num_samples/self.num_latlon), self.num_latlon, 60))
            netsw = output[:,120].reshape((int(num_samples/self.num_latlon), self.num_latlon))
            flwds = output[:,121].reshape((int(num_samples/self.num_latlon), self.num_latlon))
            precsc = output[:,122].reshape((int(num_samples/self.num_latlon), self.num_latlon))
            precc = output[:,123].reshape((int(num_samples/self.num_latlon), self.num_latlon))
            sols = output[:,124].reshape((int(num_samples/self.num_latlon), self.num_latlon))
            soll = output[:,125].reshape((int(num_samples/self.num_latlon), self.num_latlon))
            solsd = output[:,126].reshape((int(num_samples/self.num_latlon), self.num_latlon))
            solld = output[:,127].reshape((int(num_samples/self.num_latlon), self.num_latlon))
            if just_weights:
                ptend_t_weight = weightings[:,:60].reshape((int(num_samples/self.num_latlon), self.num_latlon, 60))
                ptend_q0001_weight = weightings[:,60:120].reshape((int(num_samples/self.num_latlon), self.num_latlon, 60))
                netsw_weight = weightings[:,120].reshape((int(num_samples/self.num_latlon), self.num_latlon))
                flwds_weight = weightings[:,121].reshape((int(num_samples/self.num_latlon), self.num_latlon))
                precsc_weight = weightings[:,122].reshape((int(num_samples/self.num_latlon), self.num_latlon))
                precc_weight = weightings[:,123].reshape((int(num_samples/self.num_latlon), self.num_latlon))
                sols_weight = weightings[:,124].reshape((int(num_samples/self.num_latlon), self.num_latlon))
                soll_weight = weightings[:,125].reshape((int(num_samples/self.num_latlon), self.num_latlon))
                solsd_weight = weightings[:,126].reshape((int(num_samples/self.num_latlon), self.num_latlon))
                solld_weight = weightings[:,127].reshape((int(num_samples/self.num_latlon), self.num_latlon))
        else:
            ptend_t = output[:,:60].reshape((int(num_samples/self.num_latlon), self.num_latlon, 60))
            ptend_q0001 = output[:,60:120].reshape((int(num_samples/self.num_latlon), self.num_latlon, 60))
            ptend_q0002 = output[:,120:180].reshape((int(num_samples/self.num_latlon), self.num_latlon, 60))
            ptend_q0003 = output[:,180:240].reshape((int(num_samples/self.num_latlon), self.num_latlon, 60))
            ptend_u = output[:,240:300].reshape((int(num_samples/self.num_latlon), self.num_latlon, 60))
            ptend_v = output[:,300:360].reshape((int(num_samples/self.num_latlon), self.num_latlon, 60))
            netsw = output[:,360].reshape((int(num_samples/self.num_latlon), self.num_latlon))
            flwds = output[:,361].reshape((int(num_samples/self.num_latlon), self.num_latlon))
            precsc = output[:,362].reshape((int(num_samples/self.num_latlon), self.num_latlon))
            precc = output[:,363].reshape((int(num_samples/self.num_latlon), self.num_latlon))
            sols = output[:,364].reshape((int(num_samples/self.num_latlon), self.num_latlon))
            soll = output[:,365].reshape((int(num_samples/self.num_latlon), self.num_latlon))
            solsd = output[:,366].reshape((int(num_samples/self.num_latlon), self.num_latlon))
            solld = output[:,367].reshape((int(num_samples/self.num_latlon), self.num_latlon))
            state_wind = ((ptend_u**2) + (ptend_v**2))**.5
            self.target_energy_conv['ptend_wind'] = state_wind
            if just_weights:
                ptend_t_weight = weightings[:,:60].reshape((int(num_samples/self.num_latlon), self.num_latlon, 60))
                ptend_q0001_weight = weightings[:,60:120].reshape((int(num_samples/self.num_latlon), self.num_latlon, 60))
                ptend_q0002_weight = weightings[:,120:180].reshape((int(num_samples/self.num_latlon), self.num_latlon, 60))
                ptend_q0003_weight = weightings[:,180:240].reshape((int(num_samples/self.num_latlon), self.num_latlon, 60))
                ptend_u_weight = weightings[:,240:300].reshape((int(num_samples/self.num_latlon), self.num_latlon, 60))
                ptend_v_weight = weightings[:,300:360].reshape((int(num_samples/self.num_latlon), self.num_latlon, 60))
                netsw_weight = weightings[:,360].reshape((int(num_samples/self.num_latlon), self.num_latlon))
                flwds_weight = weightings[:,361].reshape((int(num_samples/self.num_latlon), self.num_latlon))
                precsc_weight = weightings[:,362].reshape((int(num_samples/self.num_latlon), self.num_latlon))
                precc_weight = weightings[:,363].reshape((int(num_samples/self.num_latlon), self.num_latlon))
                sols_weight = weightings[:,364].reshape((int(num_samples/self.num_latlon), self.num_latlon))
                soll_weight = weightings[:,365].reshape((int(num_samples/self.num_latlon), self.num_latlon))
                solsd_weight = weightings[:,366].reshape((int(num_samples/self.num_latlon), self.num_latlon))
                solld_weight = weightings[:,367].reshape((int(num_samples/self.num_latlon), self.num_latlon))
            
        # ptend_t = ptend_t.transpose((2,0,1))
        # ptend_q0001 = ptend_q0001.transpose((2,0,1))
        # scalar_outputs = scalar_outputs.transpose((2,0,1))

        # [0] Undo output scaling
        if self.normalize:
            ptend_t = ptend_t/self.output_scale['ptend_t'].values[np.newaxis, np.newaxis, :]
            ptend_q0001 = ptend_q0001/self.output_scale['ptend_q0001'].values[np.newaxis, np.newaxis, :]
            netsw = netsw/self.output_scale['cam_out_NETSW'].values
            flwds = flwds/self.output_scale['cam_out_FLWDS'].values
            precsc = precsc/self.output_scale['cam_out_PRECSC'].values
            precc = precc/self.output_scale['cam_out_PRECC'].values
            sols = sols/self.output_scale['cam_out_SOLS'].values
            soll = soll/self.output_scale['cam_out_SOLL'].values
            solsd = solsd/self.output_scale['cam_out_SOLSD'].values
            solld = solld/self.output_scale['cam_out_SOLLD'].values
            if just_weights:
                ptend_t_weight = ptend_t_weight/self.output_scale['ptend_t'].values[np.newaxis, np.newaxis, :]
                ptend_q0001_weight = ptend_q0001_weight/self.output_scale['ptend_q0001'].values[np.newaxis, np.newaxis, :]
                netsw_weight = netsw_weight/self.output_scale['cam_out_NETSW'].values
                flwds_weight = flwds_weight/self.output_scale['cam_out_FLWDS'].values
                precsc_weight = precsc_weight/self.output_scale['cam_out_PRECSC'].values
                precc_weight = precc_weight/self.output_scale['cam_out_PRECC'].values
                sols_weight = sols_weight/self.output_scale['cam_out_SOLS'].values
                soll_weight = soll_weight/self.output_scale['cam_out_SOLL'].values
                solsd_weight = solsd_weight/self.output_scale['cam_out_SOLSD'].values
                solld_weight = solld_weight/self.output_scale['cam_out_SOLLD'].values
            if self.full_vars:
                ptend_q0002 = ptend_q0002/self.output_scale['ptend_q0002'].values[np.newaxis, np.newaxis, :]
                ptend_q0003 = ptend_q0003/self.output_scale['ptend_q0003'].values[np.newaxis, np.newaxis, :]
                ptend_u = ptend_u/self.output_scale['ptend_u'].values[np.newaxis, np.newaxis, :]
                ptend_v = ptend_v/self.output_scale['ptend_v'].values[np.newaxis, np.newaxis, :]
                if just_weights:
                    ptend_q0002_weight = ptend_q0002_weight/self.output_scale['ptend_q0002'].values[np.newaxis, np.newaxis, :]
                    ptend_q0003_weight = ptend_q0003_weight/self.output_scale['ptend_q0003'].values[np.newaxis, np.newaxis, :]
                    ptend_u_weight = ptend_u_weight/self.output_scale['ptend_u'].values[np.newaxis, np.newaxis, :]
                    ptend_v_weight = ptend_v_weight/self.output_scale['ptend_v'].values[np.newaxis, np.newaxis, :]

        # [1] Weight vertical levels by dp/g
        # only for vertically-resolved variables, e.g. ptend_{t,q0001}
        # dp/g = -\rho * dz

        dp = None
        if data_split == 'train':
            dp = self.dp_train
        elif data_split == 'val':
            dp = self.dp_val
        elif data_split == 'scoring':
            dp = self.dp_scoring
        elif data_split == 'test':
            dp = self.dp_test
        assert dp is not None
        ptend_t = ptend_t * dp/self.grav
        ptend_q0001 = ptend_q0001 * dp/self.grav
        if just_weights:
            ptend_t_weight = ptend_t_weight * dp/self.grav
            ptend_q0001_weight = ptend_q0001_weight * dp/self.grav
        if self.full_vars:
            ptend_q0002 = ptend_q0002 * dp/self.grav
            ptend_q0003 = ptend_q0003 * dp/self.grav
            ptend_u = ptend_u * dp/self.grav
            ptend_v = ptend_v * dp/self.grav
            if just_weights:
                ptend_q0002_weight = ptend_q0002_weight * dp/self.grav
                ptend_q0003_weight = ptend_q0003_weight * dp/self.grav
                ptend_u_weight = ptend_u_weight * dp/self.grav  
                ptend_v_weight = ptend_v_weight * dp/self.grav

        # [2] weight by area

        ptend_t = ptend_t * self.area_wgt[np.newaxis, :, np.newaxis]
        ptend_q0001 = ptend_q0001 * self.area_wgt[np.newaxis, :, np.newaxis]
        netsw = netsw * self.area_wgt[np.newaxis, :]
        flwds = flwds * self.area_wgt[np.newaxis, :]
        precsc = precsc * self.area_wgt[np.newaxis, :]
        precc = precc * self.area_wgt[np.newaxis, :]
        sols = sols * self.area_wgt[np.newaxis, :]
        soll = soll * self.area_wgt[np.newaxis, :]
        solsd = solsd * self.area_wgt[np.newaxis, :]
        solld = solld * self.area_wgt[np.newaxis, :]
        if just_weights:
            ptend_t_weight = ptend_t_weight * self.area_wgt[np.newaxis, :, np.newaxis]
            ptend_q0001_weight = ptend_q0001_weight * self.area_wgt[np.newaxis, :, np.newaxis]
            netsw_weight = netsw_weight * self.area_wgt[np.newaxis, :]
            flwds_weight = flwds_weight * self.area_wgt[np.newaxis, :]
            precsc_weight = precsc_weight * self.area_wgt[np.newaxis, :]
            precc_weight = precc_weight * self.area_wgt[np.newaxis, :]
            sols_weight = sols_weight * self.area_wgt[np.newaxis, :]
            soll_weight = soll_weight * self.area_wgt[np.newaxis, :]
            solsd_weight = solsd_weight * self.area_wgt[np.newaxis, :]
            solld_weight = solld_weight * self.area_wgt[np.newaxis, :]
        if self.full_vars:
            ptend_q0002 = ptend_q0002 * self.area_wgt[np.newaxis, :, np.newaxis]
            ptend_q0003 = ptend_q0003 * self.area_wgt[np.newaxis, :, np.newaxis]
            ptend_u = ptend_u * self.area_wgt[np.newaxis, :, np.newaxis]
            ptend_v = ptend_v * self.area_wgt[np.newaxis, :, np.newaxis]
            if just_weights:
                ptend_q0002_weight = ptend_q0002_weight * self.area_wgt[np.newaxis, :, np.newaxis]
                ptend_q0003_weight = ptend_q0003_weight * self.area_wgt[np.newaxis, :, np.newaxis]
                ptend_u_weight = ptend_u_weight * self.area_wgt[np.newaxis, :, np.newaxis]
                ptend_v_weight = ptend_v_weight * self.area_wgt[np.newaxis, :, np.newaxis]

        # [3] unit conversion

        ptend_t = ptend_t * self.target_energy_conv['ptend_t']
        ptend_q0001 = ptend_q0001 * self.target_energy_conv['ptend_q0001']
        netsw = netsw * self.target_energy_conv['cam_out_NETSW']
        flwds = flwds * self.target_energy_conv['cam_out_FLWDS']
        precsc = precsc * self.target_energy_conv['cam_out_PRECSC']
        precc = precc * self.target_energy_conv['cam_out_PRECC']
        sols = sols * self.target_energy_conv['cam_out_SOLS']
        soll = soll * self.target_energy_conv['cam_out_SOLL']
        solsd = solsd * self.target_energy_conv['cam_out_SOLSD']
        solld = solld * self.target_energy_conv['cam_out_SOLLD']
        if just_weights:
            ptend_t_weight = ptend_t_weight * self.target_energy_conv['ptend_t']
            ptend_q0001_weight = ptend_q0001_weight * self.target_energy_conv['ptend_q0001']
            netsw_weight = netsw_weight * self.target_energy_conv['cam_out_NETSW']
            flwds_weight = flwds_weight * self.target_energy_conv['cam_out_FLWDS']
            precsc_weight = precsc_weight * self.target_energy_conv['cam_out_PRECSC']
            precc_weight = precc_weight * self.target_energy_conv['cam_out_PRECC']
            sols_weight = sols_weight * self.target_energy_conv['cam_out_SOLS']
            soll_weight = soll_weight * self.target_energy_conv['cam_out_SOLL']
            solsd_weight = solsd_weight * self.target_energy_conv['cam_out_SOLSD']
            solld_weight = solld_weight * self.target_energy_conv['cam_out_SOLLD']
        if self.full_vars:
            ptend_q0002 = ptend_q0002 * self.target_energy_conv['ptend_q0002']
            ptend_q0003 = ptend_q0003 * self.target_energy_conv['ptend_q0003']
            ptend_u = ptend_u * self.target_energy_conv['ptend_wind']
            ptend_v = ptend_v * self.target_energy_conv['ptend_wind']
            if just_weights:
                ptend_q0002_weight = ptend_q0002_weight * self.target_energy_conv['ptend_q0002']
                ptend_q0003_weight = ptend_q0003_weight * self.target_energy_conv['ptend_q0003']
                ptend_u_weight = ptend_u_weight * self.target_energy_conv['ptend_wind']
                ptend_v_weight = ptend_v_weight * self.target_energy_conv['ptend_wind']


        if just_weights:
            if self.full_vars:
                weightings = np.concatenate([ptend_t_weight.reshape((num_samples, 60)), \
                                             ptend_q0001_weight.reshape((num_samples, 60)), \
                                             ptend_q0002_weight.reshape((num_samples, 60)), \
                                             ptend_q0003_weight.reshape((num_samples, 60)), \
                                             ptend_u_weight.reshape((num_samples, 60)), \
                                             ptend_v_weight.reshape((num_samples, 60)), \
                                             netsw_weight.reshape((num_samples))[:, np.newaxis], \
                                             flwds_weight.reshape((num_samples))[:, np.newaxis], \
                                             precsc_weight.reshape((num_samples))[:, np.newaxis], \
                                             precc_weight.reshape((num_samples))[:, np.newaxis], \
                                             sols_weight.reshape((num_samples))[:, np.newaxis], \
                                             soll_weight.reshape((num_samples))[:, np.newaxis], \
                                             solsd_weight.reshape((num_samples))[:, np.newaxis], \
                                             solld_weight.reshape((num_samples))[:, np.newaxis]], axis = 1)
            else:
                weightings = np.concatenate([ptend_t_weight.reshape((num_samples, 60)), \
                                             ptend_q0001_weight.reshape((num_samples, 60)), \
                                             netsw_weight.reshape((num_samples))[:, np.newaxis], \
                                             flwds_weight.reshape((num_samples))[:, np.newaxis], \
                                             precsc_weight.reshape((num_samples))[:, np.newaxis], \
                                             precc_weight.reshape((num_samples))[:, np.newaxis], \
                                             sols_weight.reshape((num_samples))[:, np.newaxis], \
                                             soll_weight.reshape((num_samples))[:, np.newaxis], \
                                             solsd_weight.reshape((num_samples))[:, np.newaxis], \
                                             solld_weight.reshape((num_samples))[:, np.newaxis]], axis = 1)
            return weightings
        else:
            var_dict = {'ptend_t':ptend_t,
                        'ptend_q0001':ptend_q0001,
                        'cam_out_NETSW':netsw,
                        'cam_out_FLWDS':flwds,
                        'cam_out_PRECSC':precsc,
                        'cam_out_PRECC':precc,
                        'cam_out_SOLS':sols,
                        'cam_out_SOLL':soll,
                        'cam_out_SOLSD':solsd,
                        'cam_out_SOLLD':solld}
            if self.full_vars:
                var_dict['ptend_q0002'] = ptend_q0002
                var_dict['ptend_q0003'] = ptend_q0003
                var_dict['ptend_u'] = ptend_u
                var_dict['ptend_v'] = ptend_v

            return var_dict

    def reweight_target(self, data_split):
        '''
        data_split should be train, val, scoring, or test
        weights target variables assuming V1 outputs using the output_weighting function
        '''
        assert data_split in ['train', 'val', 'scoring', 'test'], 'Provided data_split is not valid. Available options are train, val, scoring, and test.'
        if data_split == 'train':
            assert self.target_train is not None
            self.target_weighted_train = self.output_weighting(self.target_train, data_split)
        elif data_split == 'val':
            assert self.target_val is not None
            self.target_weighted_val = self.output_weighting(self.target_val, data_split)
        elif data_split == 'scoring':
            assert self.target_scoring is not None
            self.target_weighted_scoring = self.output_weighting(self.target_scoring, data_split)
        elif data_split == 'test':
            assert self.target_test is not None
            self.target_weighted_test = self.output_weighting(self.target_test, data_split)