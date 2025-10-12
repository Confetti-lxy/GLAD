from . import BaseActor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
import torch

from ...utils.heapmap_utils import generate_heatmap


class GLADActor(BaseActor):
    """ Actor for training GLAD"""
    def __init__(self, net, objective, loss_weight, settings):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize  # batch size
        self.run_score_head = False

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'template', 'search', 'gt_bbox'.
            template_images: (N_t, batch, 3, H, W)
            search_images: (N_s, batch, 3, H, W)
        returns:
            loss    - the training loss
            status  -  dict containing detailed losses
        """
        # forward pass
        out_dict = self.forward_pass(data, run_score_head=self.run_score_head)

        # process the groundtruth
        gt_bboxes = data['search_anno']  # (Ns, batch, 4) (x1,y1,w,h)

        labels = None
        if 'pred_scores' in out_dict:
            try:
                labels = data['label'].view(-1)  # (batch, ) 0 or 1
            except:
                raise Exception("Please setting proper labels for score branch.")

        # compute losses
        loss, status = self.compute_losses(out_dict, gt_bboxes, labels=labels)

        return loss, status

    def forward_pass(self, data, run_score_head):
        search_bboxes = box_xywh_to_xyxy(data['search_anno'].clone())
        # out_dict, _ = self.net(data['template_images'][0], data['search_images'],
        #                        run_score_head=run_score_head, gt_bboxes=search_bboxes)

        out_dict, _ = self.net(data['template_images'][0], data['search_images'][0], data['text'])

        # # Visualize SD
        # self.net.forward_vis_sd(data['template_images'][0], data['search_images'][0], data['text'])
        
        # out_dict: (B, N, C), outputs_coord: (1, B, N, C), target_query: (1, B, N, C)
        return out_dict

    def compute_losses(self, pred_dict, gt_bbox, return_status=True, labels=None):
        # gt gaussian map
        if self.settings.output_sz['search'] == 336:
            stride = 14
        else:
            stride = 16
        gt_gaussian_maps = generate_heatmap(gt_bbox, self.settings.output_sz['search'], stride)  # e.g., search.size=256, stride=16
        gt_gaussian_maps = gt_gaussian_maps[-1].unsqueeze(1)

        gt_bbox = gt_bbox[-1]

        # Get boxes
        pred_boxes = pred_dict['pred_boxes']
        if torch.isnan(pred_boxes).any():
            # TODO: bug fix! Need to remove NAN from training!

            # Solution1 - check nan value and set to 0, not work
            # is_nan = torch.isnan(pred_boxes)
            # pred_boxes[is_nan] = 0.0
            # print(f"Network outputs is NAN on {pred_boxes.device}! NAN mask is: ")
            # print(is_nan)
            # print("Pred boxes are: ")
            # print(pred_boxes)
            # print("Setting NAN to 0 to continue training!")

            # Solution2 - set loss to 0, not work
            # print("Network outputs is NAN! Set loss to 0.0 to continue training!")
            # if return_status:
            #     if 'score_map' in pred_dict:
            #         status = {"Loss/total": 0.0,
            #                   "Loss/ciou": 0.0,
            #                   "Loss/l1": 0.0,
            #                   "Loss/location": 0.0,
            #                   "IoU": 0.0}
            #     else:
            #         status = {"Loss/total": 0.0,
            #                   "Loss/ciou": 0.0,
            #                   "Loss/l1": 0.0,
            #                   "IoU": 0.0}
            #     return torch.tensor(0.0).cuda(), status
            # else:
            #     return torch.tensor(0.0).cuda()

            # Solution3 - raise error and exit
            raise ValueError("Network outputs is NAN! Stop Training")
        num_queries = pred_boxes.size(1)
        pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
        gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0, max=1.0)  # (B,4) --> (B,1,4) --> (B,N,4)
        # compute ciou and iou
        try:
            ciou_loss, iou = self.objective['ciou'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        except:
            ciou_loss, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
        # compute l1 loss
        l1_loss = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)

        # compute location loss
        if 'score_map' in pred_dict:
            location_loss = self.objective['focal'](pred_dict['score_map'], gt_gaussian_maps)
        else:
            location_loss = torch.tensor(0.0, device=l1_loss.device)

        # weighted sum
        # loss = self.loss_weight['ciou'] * ciou_loss + self.loss_weight['l1'] * l1_loss
        loss = self.loss_weight['ciou'] * ciou_loss + self.loss_weight['l1'] * l1_loss + \
               self.loss_weight['focal'] * location_loss

        # compute cls loss if neccessary
        if 'pred_scores' in pred_dict:
            score_loss = self.objective['score'](pred_dict['pred_scores'].view(-1), labels)
            loss = score_loss * self.loss_weight['score']

        if return_status:
            # status for log
            mean_iou = iou.detach().mean()
            if 'pred_scores' in pred_dict:
                status = {"Loss/total": loss.item(),
                          "Loss/scores": score_loss.item()}
                # status = {"Loss/total": loss.item(),
                #           "Loss/scores": score_loss.item(),
                #           "Loss/ciou": ciou_loss.item(),
                #           "Loss/l1": l1_loss.item(),
                #           "IoU": mean_iou.item()}
            elif 'score_map' in pred_dict:
                status = {"Loss/total": loss.item(),
                          "Loss/ciou": ciou_loss.item(),
                          "Loss/l1": l1_loss.item(),
                          "Loss/location": location_loss.item(),
                          "IoU": mean_iou.item()}
            else:
                status = {"Loss/total": loss.item(),
                          "Loss/ciou": ciou_loss.item(),
                          "Loss/l1": l1_loss.item(),
                          "IoU": mean_iou.item()}
            return loss, status
        else:
            return loss
