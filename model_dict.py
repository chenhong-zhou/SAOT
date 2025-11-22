from model import SAOT_Structured_Mesh_2D,SAOT_Irregular_Mesh 



def get_model(args):
    model_dict = {
        'SAOT_Structured_Mesh_2D': SAOT_Structured_Mesh_2D,
        'SAOT_Irregular_Mesh': SAOT_Irregular_Mesh
    }
    return model_dict[args.model]
