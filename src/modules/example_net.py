import torch
import torch.nn.init as init

class ExampleNet(torch.nn.Module):
    """

    Args:

    """



    def __init__(self,dim_embeddings,similarity='inner_product'):
        super(ExampleNet,self).__init__()
        self.context_lstm = torch.nn.LSTM(
            input_size = dim_embeddings,
            hidden_size = 256,
            num_layers = 1,
            batch_first = True,
            #if set bidirectional, output size will become hidden_size * 2
            bidirectional = True
        )
        #linear(input_size,output_size)
        self.linear = torch.nn.Linear(512,512)
        self.interaction_lstm = torch.nn.LSTM(
            input_size = 2048,
            hidden_size = 256,
            num_layers = 1,
            batch_first = True,
            bidirectional = True
        )


    def forward(self, context, context_lens, options, option_lens):
        '''
        context.size(): 10 * x (max context_len) * 300
        options.size(): 10 * 5(n_samples), x (max options_len), 300
        options.transpose(1, 0).size(): 5 * 10 * x(max option_len) * 300
        context_lens: A int list, batch = 10, so ten sentences in
                     and each represents the sentence length.
        option_lens: A int 2-dim list. Each list has 5(n_samples) and 10 lists.
        '''
        

        #(10,context_length,256) 256 nums, context_total_words a group , 10 these groups.
        '''
        not sure if relu have to be set here.
        u_out = torch.nn.functional.relu(u[:,-1,:])
        only get the last sentence of a context.
        narrow it to 10*1*256
        ''' 
        u_out,(h_n,h_c) = self.context_lstm(context,None)
        logits = []
        #make c into (10,256,5)
        for i,option in enumerate(options.transpose(1,0)):
            c_out,(h_n,h_c) = self.context_lstm(option,None)
            #c_out.size = (10,option_lens,256)
            energy = torch.bmm(u_out,c_out.transpose(2,1))
            #(10,context_len,option_len)
            energy = torch.nn.functional.softmax(energy,dim = 1)
            # energy = torch.nn.functional.softmax(\
            # energy.view(-1,u_out.size(1)),dim = 1).view(u_out.size(0),-1,u_out.size(1)).transpose(1,2)
            # energy (10,context_len,option_len)
            attn_u_vec = torch.bmm(energy.transpose(1,2),u_out)
            # attn_c_vec = torch.bmm(energy.transpose(1,2),u_out)
            # attn_vec (10,context_len,256)
            # 3 c_out mix
            att_and_uout = torch.cat((c_out, attn_u_vec, c_out * attn_u_vec, c_out - attn_u_vec),2)

            att_and_uout = self.interaction_lstm(att_and_uout,None)[0].max(1)[0]
            att_and_uout = self.linear(att_and_uout)

            att_and_uout = torch.tanh(att_and_uout)

            logits.append(att_and_uout)
            # interaction = (10,context_len)
        logits = torch.stack(logits,2)
        u_out = u_out.max(1)[0]
        u_out = u_out.view(u_out.size(0),1,u_out.size(1))
        logits = torch.bmm(u_out,logits)
        logits = torch.squeeze(logits)
        return logits

    # def showAttention(input_sentence, output_words, attentions):
    #     # Set up figure with colorbar
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111)
    #     cax = ax.matshow(attentions.numpy(), cmap='bone')
    #     fig.colorbar(cax)

    #     # Set up axes
    #     ax.set_xticklabels([''] + input_sentence.split(' ') +
    #                        ['[SEP]'], rotation=90)
    #     ax.set_yticklabels([''] + output_words)

    #     # Show label at every tick
    #     ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    #     ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    #     plt.show()
    # def evaluateAndShowAttention(input_sentence):
    #     output_words, attentions = evaluate(
    #         encoder1, attn_decoder1, input_sentence)
    #     print('input =', input_sentence)
    #     print('output =', ' '.join(output_words))
    #     showAttention(input_sentence, output_words, attentions)

