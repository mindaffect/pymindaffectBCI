# import the most common user classes
__all__=['utopiaclient.UtopiaClient'
         'utopiaController.UtopiaController'
         'noisetag.Noisetag'
         'online_bci']

if __name__=="__main__":
    import mindaffectBCI.online_bci
    args = mindaffectBCI.online_bci.parse_args()
    mindaffectBCI.online_bci.run(label=args.label, logdir=args.logdir, acquisition=args.acquisition, acq_args=args.acq_args, 
                   decoder=args.decoder, decoder_args=args.decoder_args, 
                   presentation=args.presentation, presentation_args=args.presentation_args)