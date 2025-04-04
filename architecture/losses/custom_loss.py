class HierarchicalDetectionLoss(nn.Module):
    def __init__(self, lambda_box=1.0, lambda_cls=1.0, lambda_staff=2.0, lambda_rel=0.5):
        super().__init__()
        self.lambda_box = lambda_box
        self.lambda_cls = lambda_cls
        self.lambda_staff = lambda_staff
        self.lambda_rel = lambda_rel
        
        # Use Focal Loss for classification to handle class imbalance
        self.cls_loss = FocalLoss(alpha=0.25, gamma=2.0)
        self.box_loss = nn.SmoothL1Loss()
        
        # Staff completeness loss
        self.staff_loss = StaffCompletenessLoss()
        
        # Relationship consistency loss
        self.relationship_loss = RelationshipConsistencyLoss()
        
    def forward(self, predictions, targets):
        # Standard detection losses
        box_loss = self.box_loss(predictions['boxes'], targets['boxes'])
        cls_loss = self.cls_loss(predictions['scores'], targets['classes'])
        
        # Staff completeness loss - ensures each staff has 5 lines
        staff_loss = self.staff_loss(predictions['stafflines'], targets['stafflines'])
        
        # Relationship consistency loss
        rel_loss = self.relationship_loss(predictions['relationships'], targets['relationships'])
        
        # Combined loss
        total_loss = self.lambda_box * box_loss + \
                     self.lambda_cls * cls_loss + \
                     self.lambda_staff * staff_loss + \
                     self.lambda_rel * rel_loss
        
        return total_loss, {
            'box_loss': box_loss,
            'cls_loss': cls_loss,
            'staff_loss': staff_loss,
            'rel_loss': rel_loss
        }