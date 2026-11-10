# exportador_onnx.py - V8 (com instalação automática de dependências)
import os
import torch
import numpy as np
import subprocess
import sys
from rede import DQN
from config import CHECKPOINT_DIR

class ExportadorONNX:
    def __init__(self, checkpoint_dir=None):
        self.checkpoint_dir = checkpoint_dir or CHECKPOINT_DIR
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_dim = None
        self.model = None
        
        print(f"🔧 Exportador ONNX inicializado")
        print(f"   Diretório de checkpoints: {self.checkpoint_dir}")
        print(f"   Dispositivo: {self.device}")
    
    def detectar_dimensao_entrada(self, checkpoint_path):
        """Carrega o checkpoint para detectar a dimensão da entrada."""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            if 'policy_state_dict' in checkpoint:
                # Pega o shape da primeira camada linear (fc1.weight)
                weights = checkpoint['policy_state_dict']['fc1.weight']
                # weights.shape é [hidden, input_dim]
                input_dim = weights.shape[1]
                return input_dim
        except Exception as e:
            print(f"❌ Erro ao detectar dimensão: {e}")
            return None
    
    def encontrar_ultimo_checkpoint(self):
        if not os.path.exists(self.checkpoint_dir):
            print(f"❌ Pasta não encontrada: {self.checkpoint_dir}")
            return None
        
        arquivos = [f for f in os.listdir(self.checkpoint_dir) if f.endswith('.pth')]
        if not arquivos:
            print(f"❌ Nenhum checkpoint encontrado em {self.checkpoint_dir}")
            return None
        
        # Prioridade: latest > final > ep (mais recente)
        if 'checkpoint_latest.pth' in arquivos:
            return os.path.join(self.checkpoint_dir, 'checkpoint_latest.pth')
        elif 'checkpoint_final.pth' in arquivos:
            return os.path.join(self.checkpoint_dir, 'checkpoint_final.pth')
        else:
            ep_arquivos = [f for f in arquivos if f.startswith('checkpoint_ep')]
            if ep_arquivos:
                nums = []
                for f in ep_arquivos:
                    try:
                        num = int(f.replace('checkpoint_ep', '').replace('.pth', ''))
                        nums.append((num, f))
                    except:
                        pass
                if nums:
                    nums.sort(key=lambda x: x[0], reverse=True)
                    return os.path.join(self.checkpoint_dir, nums[0][1])
        
        return os.path.join(self.checkpoint_dir, arquivos[0])
    
    def carregar_checkpoint(self, caminho):
        if not os.path.exists(caminho):
            return False
        
        try:
            # Detecta dimensão da entrada
            self.input_dim = self.detectar_dimensao_entrada(caminho)
            if self.input_dim is None:
                return False
            print(f"🔍 Dimensão de entrada detectada: {self.input_dim}")
            
            # Cria o modelo com a dimensão correta
            from rede import DQN
            self.model = DQN(input_dim=self.input_dim, output_dim=4).to(self.device)
            self.model.eval()
            
            checkpoint = torch.load(caminho, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['policy_state_dict'])
            print(f"✅ Checkpoint carregado: {caminho}")
            print(f"   Episódio: {checkpoint.get('episode', 'desconhecido')}")
            print(f"   Epsilon: {checkpoint.get('epsilon', 'desconhecido')}")
            print(f"   Entradas: {self.input_dim}")
            return True
        except Exception as e:
            print(f"❌ Erro ao carregar checkpoint: {e}")
            return False
    
    def exportar_onnx(self, output_path):
        if self.model is None:
            print("❌ Nenhum modelo carregado.")
            return False
        
        # Cria entrada de exemplo
        exemplo_entrada = torch.randn(1, self.input_dim, device=self.device)
        input_names = ['observacao']
        output_names = ['q_values']
        
        try:
            torch.onnx.export(
                self.model,
                exemplo_entrada,
                output_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes={
                    'observacao': {0: 'batch_size'},
                    'q_values': {0: 'batch_size'}
                }
            )
            print(f"✅ Modelo exportado para ONNX: {output_path}")
            print(f"   Entrada: {input_names} (batch x {self.input_dim})")
            print(f"   Saída: {output_names} (batch x 4)")
            if os.path.exists(output_path):
                tamanho = os.path.getsize(output_path) / (1024 * 1024)
                print(f"   Tamanho: {tamanho:.2f} MB")
            return True
        except Exception as e:
            print(f"❌ Erro ao exportar: {e}")
            if "No module named 'onnxscript'" in str(e):
                print("\n💡 Instalando onnxscript automaticamente...")
                try:
                    subprocess.check_call([sys.executable, "-m", "pip", "install", "onnxscript"])
                    print("✅ onnxscript instalado. Tente novamente.")
                except Exception as install_error:
                    print(f"❌ Falha ao instalar onnxscript: {install_error}")
                    print("   Instale manualmente: pip install onnxscript")
            return False
    
    def exportar_ultimo_checkpoint(self, output_path=None):
        checkpoint_path = self.encontrar_ultimo_checkpoint()
        if checkpoint_path is None:
            return False
        
        print(f"\n📂 Checkpoint encontrado: {checkpoint_path}")
        
        if not self.carregar_checkpoint(checkpoint_path):
            return False
        
        if output_path is None:
            nome_base = os.path.splitext(os.path.basename(checkpoint_path))[0]
            output_path = os.path.join(self.checkpoint_dir, f"{nome_base}.onnx")
        
        return self.exportar_onnx(output_path)
    
    def testar_modelo_onnx(self, onnx_path):
        try:
            import onnxruntime as ort
            import numpy as np
            sessao = ort.InferenceSession(onnx_path)
            entrada_teste = np.random.randn(1, self.input_dim).astype(np.float32)
            resultados = sessao.run(None, {'observacao': entrada_teste})
            print(f"\n🧪 Teste do modelo ONNX:")
            print(f"   Entrada: {entrada_teste.shape}")
            print(f"   Saída (Q-values): {resultados[0]}")
            print(f"   Ação escolhida: {np.argmax(resultados[0])}")
            return True
        except ImportError:
            print("\n⚠️  onnxruntime não está instalado.")
            print("   Para testar: pip install onnxruntime")
            return False
        except Exception as e:
            print(f"\n❌ Erro ao testar: {e}")
            return False


# =========================================================
# EXECUÇÃO DIRETA
# =========================================================
if __name__ == "__main__":
    print("="*60)
    print("🔧 EXPORTADOR ONNX - V8")
    print("="*60)
    
    exportador = ExportadorONNX()
    sucesso = exportador.exportar_ultimo_checkpoint()
    
    if sucesso:
        import glob
        arquivos_onnx = glob.glob(os.path.join(exportador.checkpoint_dir, "*.onnx"))
        if arquivos_onnx:
            arquivo_onnx = max(arquivos_onnx, key=os.path.getmtime)
            exportador.testar_modelo_onnx(arquivo_onnx)
    
    print("\n" + "="*60)
    print("🏁 Processo concluído!")
    print("="*60)   