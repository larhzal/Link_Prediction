import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch_geometric.data import Data
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix
import seaborn as sns

class PyTorchGeometricExtractor:
    """
    Extract predictions from your 3 PyTorch Geometric models (GCN, GAT, GraphSAGE)
    """
    
    def __init__(self):
        self.model_predictions = {}
        self.test_data = None
        self.true_labels = None
        
    def load_test_data(self, test_links_path):
        """
        Load your test_links.csv file
        """
        self.test_data = pd.read_csv(test_links_path)
        print(f"✅ Loaded test data: {len(self.test_data)} samples")
        print(f"📊 Columns: {list(self.test_data.columns)}")
        
        # Extract true labels
        if 'label' in self.test_data.columns:
            self.true_labels = self.test_data['label'].values
            pos_links = np.sum(self.true_labels)
            neg_links = len(self.true_labels) - pos_links
            print(f"📈 Positive links (1): {pos_links}")
            print(f"📉 Negative links (0): {neg_links}")
            print(f"⚖️ Class balance: {pos_links/len(self.true_labels):.2%} positive")
        
        return self.test_data
    
    def prepare_test_edges(self, test_data):
        """
        Convert test data to edge tensor format for PyTorch Geometric
        """
        # Create edge tensor [2, num_edges] format
        test_edges = torch.tensor([
            test_data['source'].values,
            test_data['target'].values
        ], dtype=torch.long)
        
        print(f"🔗 Test edges shape: {test_edges.shape}")
        print(f"🔗 Test edges sample: {test_edges[:, :5]}")
        
        return test_edges
    
    def extract_predictions_from_model(self, model, model_name, test_edges, 
                                     node_features, edge_index, device='cpu'):
        """
        Extract predictions from a trained PyTorch Geometric model
        
        Args:
            model: Your trained GNN model (GCN, GAT, or GraphSAGE)
            model_name: Name of the model ("GCN", "GAT", "GraphSAGE")
            test_edges: Test edge indices [2, num_test_edges]
            node_features: Node feature matrix [num_nodes, num_features]
            edge_index: Training graph edges [2, num_edges]
            device: 'cpu' or 'cuda'
        """
        print(f"\n🔄 Extracting predictions from {model_name}...")
        
        # Move everything to device
        model = model.to(device)
        test_edges = test_edges.to(device)
        node_features = node_features.to(device)
        edge_index = edge_index.to(device)
        
        model.eval()  # Set to evaluation mode
        
        with torch.no_grad():
            # Step 1: Get node embeddings
            print(f"  📊 Computing node embeddings...")
            if hasattr(model, 'encode'):
                # If your model has an encode method
                z = model.encode(node_features, edge_index)
            else:
                # If your model returns embeddings directly
                z = model(node_features, edge_index)
            
            print(f"  📏 Node embeddings shape: {z.shape}")
            
            # Step 2: Decode edge predictions
            print(f"  🔗 Computing edge predictions...")
            if hasattr(model, 'decode'):
                # If your model has a decode method
                logits = model.decode(z, test_edges)
            else:
                # Manual edge prediction (common approach)
                # Get source and target node embeddings
                src_embeddings = z[test_edges[0]]  # [num_test_edges, hidden_dim]
                tgt_embeddings = z[test_edges[1]]  # [num_test_edges, hidden_dim]
                
                # Compute logits (you might need to adjust this based on your model)
                # Common approaches:
                # 1. Dot product
                logits = torch.sum(src_embeddings * tgt_embeddings, dim=1)
                
                # 2. Or if your model has a final MLP layer for edge prediction
                # edge_features = torch.cat([src_embeddings, tgt_embeddings], dim=1)
                # logits = model.edge_predictor(edge_features)  # Adjust based on your model
            
            print(f"  📈 Logits shape: {logits.shape}")
            print(f"  📊 Logits range: [{logits.min():.3f}, {logits.max():.3f}]")
            
            # Step 3: Convert logits to probabilities
            probabilities = torch.sigmoid(logits)
            
            print(f"  🎯 Probabilities range: [{probabilities.min():.3f}, {probabilities.max():.3f}]")
            
            # Convert to numpy
            probabilities = probabilities.cpu().numpy()
        
        # Store predictions
        self.model_predictions[model_name] = probabilities
        
        # Quick evaluation if we have true labels
        if self.true_labels is not None:
            auc_roc = roc_auc_score(self.true_labels, probabilities)
            auc_pr = average_precision_score(self.true_labels, probabilities)
            print(f"  ✅ {model_name} AUC-ROC: {auc_roc:.4f}")
            print(f"  ✅ {model_name} AUC-PR: {auc_pr:.4f}")
        
        return probabilities
    
    def run_all_models(self, model_files, test_edges, node_features, edge_index, device='cpu'):
        """
        Run all three models and extract predictions
        
        Args:
            model_files: Dict with model file paths
                {'GCN': 'gcn_model.py', 'GAT': 'gat_model.py', 'GraphSAGE': 'graphsage_model.py'}
            test_edges: Test edge indices
            node_features: Node features
            edge_index: Graph connectivity
        """
        print("🚀 Running all models for prediction extraction...")
        
        for model_name, model_file in model_files.items():
            print(f"\n{'='*50}")
            print(f"Loading {model_name} from {model_file}")
            print(f"{'='*50}")
            
            # You'll need to load your model here
            # This depends on how your models are structured
            # Example:
            # exec(open(model_file).read())  # If models are in separate .py files
            # model = locals()[f'{model_name}_model']  # Get the model object
            
            # For now, showing the structure:
            print(f"⚠️  You need to load your {model_name} model here")
            print(f"   Example: model = load_model_from_file('{model_file}')")
            
            # Once you have the model loaded:
            # self.extract_predictions_from_model(model, model_name, test_edges, 
            #                                   node_features, edge_index, device)
    
    def create_evaluation_plots(self):
        """
        Create all evaluation plots once you have predictions from all models
        """
        if len(self.model_predictions) == 0:
            print("❌ No model predictions available!")
            return
        
        if self.true_labels is None:
            print("❌ No true labels available!")
            return
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. ROC Curves
        ax1 = axes[0, 0]
        for model_name, predictions in self.model_predictions.items():
            fpr, tpr, _ = roc_curve(self.true_labels, predictions)
            roc_auc = auc(fpr, tpr)
            ax1.plot(fpr, tpr, linewidth=2, label=f'{model_name} (AUC = {roc_auc:.3f})')
        
        ax1.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random')
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('ROC Curves')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Precision-Recall Curves
        ax2 = axes[0, 1]
        for model_name, predictions in self.model_predictions.items():
            precision, recall, _ = precision_recall_curve(self.true_labels, predictions)
            pr_auc = auc(recall, precision)
            ax2.plot(recall, precision, linewidth=2, label=f'{model_name} (AUC = {pr_auc:.3f})')
        
        baseline = np.sum(self.true_labels) / len(self.true_labels)
        ax2.axhline(y=baseline, color='k', linestyle='--', linewidth=2, 
                   label=f'Random (Precision = {baseline:.3f})')
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title('Precision-Recall Curves')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Confusion Matrix for best model
        best_model = max(self.model_predictions.keys(), 
                        key=lambda x: roc_auc_score(self.true_labels, self.model_predictions[x]))
        
        ax3 = axes[1, 0]
        y_pred_binary = (self.model_predictions[best_model] > 0.5).astype(int)
        cm = confusion_matrix(self.true_labels, y_pred_binary)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['No Link', 'Link'],
                   yticklabels=['No Link', 'Link'], ax=ax3)
        ax3.set_title(f'Confusion Matrix - {best_model}')
        ax3.set_xlabel('Predicted')
        ax3.set_ylabel('Actual')
        
        # 4. Prediction Distribution
        ax4 = axes[1, 1]
        for model_name, predictions in self.model_predictions.items():
            ax4.hist(predictions, bins=30, alpha=0.7, label=model_name)
        ax4.set_xlabel('Prediction Probability')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Prediction Distributions')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print summary
        self.print_evaluation_summary()
    
    def print_evaluation_summary(self):
        """
        Print detailed evaluation summary
        """
        print("\n" + "="*60)
        print("📊 MODEL EVALUATION SUMMARY")
        print("="*60)
        
        results = []
        for model_name, predictions in self.model_predictions.items():
            auc_roc = roc_auc_score(self.true_labels, predictions)
            auc_pr = average_precision_score(self.true_labels, predictions)
            
            binary_preds = (predictions > 0.5).astype(int)
            f1 = f1_score(self.true_labels, binary_preds)
            
            results.append({
                'Model': model_name,
                'AUC-ROC': auc_roc,
                'AUC-PR': auc_pr,
                'F1 Score': f1
            })
            
            print(f"\n🤖 {model_name}:")
            print(f"   AUC-ROC:  {auc_roc:.4f}")
            print(f"   AUC-PR:   {auc_pr:.4f}")
            print(f"   F1 Score: {f1:.4f}")
        
        # Find best model
        best_model = max(results, key=lambda x: x['AUC-ROC'])
        print(f"\n🏆 Best Model: {best_model['Model']} (AUC-ROC: {best_model['AUC-ROC']:.4f})")
        
        return results
    
    def save_predictions(self, output_path="model_predictions.csv"):
        """
        Save all predictions to CSV
        """
        if len(self.model_predictions) == 0:
            print("❌ No predictions to save!")
            return
        
        # Create DataFrame
        results_df = self.test_data.copy()
        
        # Add model predictions
        for model_name, predictions in self.model_predictions.items():
            results_df[f'{model_name}_probability'] = predictions
            results_df[f'{model_name}_prediction'] = (predictions > 0.5).astype(int)
        
        # Save
        results_df.to_csv(output_path, index=False)
        print(f"💾 Saved predictions to {output_path}")
        
        return results_df


# TEMPLATE FOR YOUR SPECIFIC USE CASE
def your_extraction_workflow():
    """
    Template workflow for your specific setup
    """
    
    print("🎯 YOUR EXTRACTION WORKFLOW")
    print("="*40)
    
    # Step 1: Initialize extractor
    extractor = PyTorchGeometricExtractor()
    
    # Step 2: Load test data
    test_data = extractor.load_test_data('test_links.csv')
    test_edges = extractor.prepare_test_edges(test_data)
    
    # Step 3: Prepare your graph data
    print("\n📋 You need to prepare these variables:")
    print("   - node_features: Your node feature matrix")
    print("   - edge_index: Your graph connectivity (training edges)")
    print("   - device: 'cpu' or 'cuda'")
    
    # Example (you'll need to adapt this):
    example_code = '''
    # Load your graph data
    node_features = torch.load('node_features.pt')  # or however you load them
    edge_index = torch.load('edge_index.pt')        # or however you load them
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load your models (adapt based on your file structure)
    import importlib.util
    
    # Load GCN model
    spec = importlib.util.spec_from_file_location("gcn_module", "gcn_model.py")
    gcn_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gcn_module)
    gcn_model = gcn_module.model  # or however your model is named
    
    # Extract predictions
    extractor.extract_predictions_from_model(
        gcn_model, "GCN", test_edges, node_features, edge_index, device
    )
    
    # Repeat for GAT and GraphSAGE...
    '''
    
    print("\n📝 Example code to adapt:")
    print(example_code)
    
    # Step 4: Create visualizations
    print("\n📊 Once you have predictions, create visualizations:")
    print("   extractor.create_evaluation_plots()")
    print("   extractor.save_predictions('results.csv')")

if __name__ == "__main__":
    your_extraction_workflow()