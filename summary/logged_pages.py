import os
from django.http import HttpResponseRedirect, HttpResponse, JsonResponse, FileResponse
from django.shortcuts import render
from .twitter import create_tweet
import datetime
from urllib.parse import unquote
import random
from summary.models import User_Model, SummaryModel, AdvancedSummary, Notifications, TranslationsModel, ApprovalModel, ContentModal, TwitterAnalysis, MeetingConversionModel
from .alogs import translate, filePDF_extract, fileDOCX_extract, fileTXT_extract, Clean_Text, PegasusModel, URL_extract, Spacy, setGrammer, Brat, parse_text_to_new_article, GenArticleHTML, fromMic, GO, gen_image_from_text, twitter_sa, speaker_diarization
from .mp3 import create_audio_transcript_file, convert_audio_to_wav
from .linkedin import post_to_linkedin
from io import StringIO, BytesIO
from math import floor

# GLOBAL VARIABLES

fileDIR = "files/"


def dashboard(request):
    if request.session.get('UserName') is None:
        return HttpResponseRedirect('login')
    else:
        userName = request.session.get('UserName')
        userDetails = userDet(userName)
        listofNotification = listNotifications(userName)
        context = {
            'userName': userName,
            'altercount': len(listofNotification),
            'notifications': listofNotification

        }

        if request.GET.get('type') == 'twitter':
            tw_id = request.GET.get('tw_id')
            tw_link = request.GET.get('tw_link')
            context['type'] = 'twitter'
            context['tw_id'] = tw_id
            context['tw_link'] = tw_link

        if request.GET.get('type') == 'linked_in':
            link_id = request.GET.get('lnk_id')
            context['type'] = 'linked_in'
            context['lnk_id'] = link_id

        if request.GET.get('approved_id'):
            context['approved_id'] = request.GET.get('approved_id')
        else:
            context['approved_id'] = None

        if request.GET.get('type'):
            context['type'] = request.GET.get('type')
        else:
            context['type'] = None

        # Dashboard Send Data
        pendingReq = ApprovalModel.objects.filter(
            approdeval_sent_by=userDetails, approvalStatus=False).count()
        context['pendingReq'] = pendingReq

        SummariesDone = SummaryModel.objects.filter(
            summary_added_by=userDetails).count()
        AdvSummariesDone = AdvancedSummary.objects.filter(
            adv_summary_added_by=userDetails).count()
        context['SummariesDone'] = SummariesDone+AdvSummariesDone

        TransDone = TranslationsModel.objects.filter(
            transDoneBy=userDetails).count()
        context['TransDone'] = TransDone

        summaryList = SummaryModel.objects.filter(
            summary_added_by=userDetails)[::-1]
        summaryList = summaryList[0:5]

        context['summaryList'] = summaryList
        context['Adv_summary'] = 0
        context['ContentGen_CNT'] = ContentModal.objects.filter(
            contentAddedBy=userDetails).count()

        return render(request, 'logged_pages/dashboard.html', context=context)


def nw_newsarticle(request):
    userName = request.session.get('UserName')
    listofNotification = listNotifications(userName)
    userDetailsfromDB = userDet(userName)
    # return HttpResponse(userDetailsfromDB)

    if request.method == 'POST':

        frm_URL_ARTICLE = request.POST.get('news_url')
        # textfromURL = []
        summaryTextPass = ""
        for i in frm_URL_ARTICLE:
            summaryTextPass = summaryTextPass + URL_extract(i)
        summary = setGrammer(
            Brat(PegasusModel(Spacy(Clean_Text(summaryTextPass))), 80, 280))
        # summary = "testing"
        char_count = summary.count('')
        word_count = summary.count(' ')
        author = "TestAuthor"
        summary_approved = False
        AddedSUMMARY = AddTableSummary(userDetailsfromDB, summary,
                                       frm_URL_ARTICLE, author)
        fileType = ''

        return HttpResponse(summaryPage(AddedSUMMARY.summary_id, summary, "", author, char_count, word_count, userName, request, summary_approved, fileType, fileURL=""))


def new_summary(request):
    ts = datetime.datetime.now()
    ts = str(int(ts.strftime("%Y%m%d%H%M%S")))
    userName = request.session.get('UserName')
    listofNotification = listNotifications(userName)

    if request.method == 'GET':
        context = {
            'userName': userName,
            'altercount': len(listofNotification),
            'notifications': listofNotification

        }
        return render(request, 'logged_pages/new_summary.html', context=context)
    else:
        fileURL = ""
        fileType = ''
        frm_uploadedFile = request.FILES.get('uploadedFile')
        lang = ""
        if request.POST.get('language'):
            lang = request.POST.get('language')

        userDir = os.path.exists(fileDIR+"/"+userName)
        test = ""
        if userDir == False:
            test = os.path.join(fileDIR, userName)
            os.mkdir(test)
        else:
            test = os.path.join(fileDIR, userName)+"/" + \
                userName+"_"+ts+"_"+frm_uploadedFile.name
            # return test
        # return HttpResponse(str(test))
        summary = ""

        handle_uploaded_file(test, frm_uploadedFile)
        file, fileEXT = os.path.splitext(test)
        if fileEXT == '.pdf':
            PDFfileContext = filePDF_extract(test)
            cleanText = Clean_Text(PDFfileContext)
            summary = setGrammer(
                PegasusModel(Spacy(Clean_Text(cleanText))))

        if fileEXT == '.doc' or fileEXT == '.docx':
            DOCXfileContent = fileDOCX_extract(test)
            cleanText = Clean_Text(DOCXfileContent)
            # summary = setGrammer(
            # PegasusModel(Spacy(Clean_Text(cleanText))))
            summary = "TEXT Lorem ad minim veniam et metus  in  hendrerit   "

        if fileEXT == '.txt':
            TXTFileContent = fileTXT_extract(test)
            cleanText = Clean_Text(TXTFileContent)
            summary = setGrammer(
                # Brat(PegasusModel(Spacy(Clean_Text(cleanText)))))
                PegasusModel(Spacy(Clean_Text(cleanText))))

        if fileEXT == '.wav' or fileEXT == '.wmv' or fileEXT == '.avi' or fileEXT == '.mp4':
            fileURL = "../"+test
            TXTFileContent = create_audio_transcript_file(
                fileURL, lang, fileEXT)

            summary = setGrammer(
                PegasusModel(Spacy(Clean_Text(TXTFileContent))))
            fileType = 'video'

        char_count = summary.count('')
        word_count = summary.count(' ')
        author = "TestAuthor"
        summary_approved = False
        userDetailsfromDB = userDet(userName)

        AddedSUMMARY = AddTableSummary(userDetailsfromDB, summary,
                                       test, author)

        return HttpResponse(summaryPage(AddedSUMMARY.summary_id, summary, "", author, char_count, word_count, userName, request, summary_approved, fileType, fileURL))


def go(request):
    ts = datetime.datetime.now()
    ts = str(int(ts.strftime("%Y%m%d%H%M%S")))
    userName = request.session.get('UserName')
    listofNotification = listNotifications(userName)
    fileURL = ''
    if request.method == 'GET':
        context = {
            'userName': userName,
            'altercount': len(listofNotification),
            'notifications': listofNotification

        }
        return render(request, 'logged_pages/fileGO.html', context=context)
    else:
        fileType = 'go'
        frm_uploadedFile = request.FILES.get('uploadedFile')
        lang = ""
        if request.POST.get('language'):
            lang = request.POST.get('language')

        userDir = os.path.exists(fileDIR+"/"+userName)
        test = ""
        if userDir == False:
            test = os.path.join(fileDIR, userName)
            os.mkdir(test)
        else:
            test = os.path.join(fileDIR, userName)+"/" + \
                userName+"_"+ts+"_"+frm_uploadedFile.name
            # return test
        # return HttpResponse(str(test))
        summary = ""

        handle_uploaded_file(test, frm_uploadedFile)
        file, fileEXT = os.path.splitext(test)
        if fileEXT == '.pdf':
            PDFfileContext = filePDF_extract(test)
            # cleanText = Clean_Text(PDFfileContext)
            summary = GO(PDFfileContext, '../'+test)

        if fileEXT == '.doc' or fileEXT == '.docx':
            DOCXfileContent = fileDOCX_extract(test)
            summary = GO(DOCXfileContent, '../'+test)

        char_count = 0
        word_count = 0
        author = "TestAuthor"
        summary_approved = False
        userDetailsfromDB = userDet(userName)

        AddedSUMMARY = AddTableSummary(userDetailsfromDB, summary[0],
                                       test, author)

        # return HttpResponse(summary)
        return HttpResponse(summaryPage(AddedSUMMARY.summary_id, summary[0], summary[1], author, char_count, word_count, userName, request, summary_approved, fileType, fileURL))

    textToReturn = GO("test")
    return HttpResponse(textToReturn)


def gen_image(request):
    ts = datetime.datetime.now()
    ts = str(int(ts.strftime("%Y%m%d%H%M%S")))
    userName = request.session.get('UserName')
    listofNotification = listNotifications(userName)
    fileURL = ''
    context = {
        'userName': userName,
        'altercount': len(listofNotification),
        'notifications': listofNotification

    }
    if request.method == 'GET':

        return render(request, 'logged_pages/gen_image.html', context=context)
    if request.method == 'POST':
        fileURL = gen_image_from_text(str(request.POST.get('text_input')))
        context['fileURL'] = fileURL
    return render(request, "logged_pages/IMG.html", context)


def nw_own_article(request):
    if request.method == 'GET':

        return render(request, "logged_pages/new_own_article.html")
    else:
        newArt = request.POST.get('newArt')
        tags = GenArticleHTML(newArt)

        context = {
            'new_art': tags.get('meta')
        }
        response = render(request, "logged_pages/down.html", context)
        file = BytesIO(response.content)
        return FileResponse(file, as_attachment=True, filename="index.html")

        # TEXT = URL_extract(request.POST.get('TEXT1'))

        # newArt = parse_text_to_new_article(request.POST.get('TEXT1'))
        # newArt = """
        # TEXT LOADE lorem ipsum dolor sit amet
        # """

        context = {
            'new_art': tags.get('text'),
            # 'new_art':newArt,
            'meta': tags.get('meta')
        }
    return render(request, "logged_pages/post_ownART.html", context)


def nw_mic(request):
    context = {}
    if request.method == 'GET':
        return render(request, "logged_pages/fromMic.html")
    else:
        lang_code = request.POST.get('language')
        userName = request.session.get('UserName')

        filename = request.FILES.get('uploadedFile')
        ts = datetime.datetime.now()
        ts = str(int(ts.strftime("%Y%m%d%H%M%S")))
        uploadFilePath = os.path.join(fileDIR, userName)+"/" + \
            userName+"_"+ts+"_"+filename.name
        handle_uploaded_file(uploadFilePath, filename)
        fileURL = "../"+uploadFilePath
        outputFinaleName = convert_audio_to_wav(fileURL)
        TXTFileContent = create_audio_transcript_file(
            outputFinaleName[1], lang_code, '.wav')
        context['returnText'] = TXTFileContent

        return render(request, "logged_pages/text_from_mic.html", context)
        # return JsonResponse(request.FILES.get('myfile',False))

# GLOBAL FUNCTIONS


def AddTableSummary(addedBy, text, org, auth):
    OBJ = SummaryModel(summary_added_by=addedBy, summary_text=text,
                       summary_org_file=org, summary_author=auth)
    OBJ.save()
    return OBJ


def handle_uploaded_file(filePassingDir, file):

    with open(filePassingDir, 'wb+') as destination:
        for chunk in file.chunks():
            destination.write(chunk)
    print("file Uploaded")


def summaryPage(summary_id, summaryText, gotables, author, char_count, word_count, userName, request, summary_approved, fileType, fileURL):
    listofNotification = listNotifications(userName)
    context = {
        'summary_id': summary_id,
        'summary': summaryText,
        'gotable': gotables,
        'AutorName': author,
        'altercount': len(listofNotification),
        'notifications': listofNotification,
        'char_count': char_count,
        'word_count': word_count,
        'userName': userName,
        'summary_approved': summary_approved,
        'fileType': fileType,
        'fileURL': fileURL

    }
    return render(request, 'logged_pages/summary_page.html', context=context)


def adv_summary(request, summary_id):
    userName = request.session.get('UserName')
    if request.method == 'GET':
        # summaryid = request.GET.get("summary_id")
        listofNotification = listNotifications(userName)
        getAdvSummaryDetails = SummaryModel.objects.filter(
            summary_id=summary_id)[0]

        SampleText = "Lorem AdvancedSummary"
        OBJ = AdvancedSummary(adv_summaryTextID=getAdvSummaryDetails, adv_summary_added_by=userDet(
            userName), adv_summary_text=SampleText)
        OBJ.save()
        adv_summary_id = OBJ.adv_summary_id
        SummaryModel.objects.filter(
            summary_id=summary_id).update(
            summary_advance_available=True, adv_summary_PK=OBJ)

        adv_author = OBJ.adv_summary_author
        adv_char_count = SampleText.count('')
        adv_word_count = SampleText.count(' ')
        adv_summary_approved = OBJ.adv_summary_approval_status

        context = {

            'adv_summary_id': adv_summary_id,
            'adv_summary_text': SampleText,
            'adv_summary_AutorName': adv_author,
            'altercount': len(listofNotification),
            'notifications': listofNotification,
            'adv_char_count': adv_char_count,
            'adv_word_count': adv_word_count,
            'userName': userName,
            'adv_summary_approved': adv_summary_approved,
            'adv_summary': SampleText

        }
        return render(request, 'logged_pages/adv_summary_page.html', context=context)
    else:
        return HttpResponse(
            "its a post"
        )


def mySummaries(request):

    userName = request.session.get('UserName')
    listofNotification = listNotifications(userName)
    ts = datetime.datetime.now()
    userDetailsfromDB = userDet(userName)

    getSummaries = SummaryModel.objects.filter(
        summary_added_by=userDetailsfromDB).all()[::-1]

    # summaryList = {
    #     'one': {'summary_id': random.randint(1, 10000), 'timestamp': ts, 'from': 'admin', 'approval_status': 'No', 'twitter': 'No', 'instagram': 'No'}

    # }
    if request.method == 'GET':
        context = {
            'userName': userName,
            'altercount': len(listofNotification),
            'notifications': listofNotification,
            'summaryList': getSummaries

        }
        return render(request, 'logged_pages/my_summaries.html', context=context)


def viewSummary(request, summary_id):
    context = {}
    userName = request.session.get('UserName')
    listofNotification = listNotifications(userName)
    getSummaryDetails = SummaryModel.objects.filter(summary_id=summary_id)[0]

    summary_approved = getSummaryDetails.summary_approval_status
    available_adv_summary = getSummaryDetails.summary_advance_available
    available_trans = getSummaryDetails.summary_availalabe_trans
    if available_trans is True:
        adv_summary_id = getSummaryDetails.adv_summary_PK.adv_summary_id
    summary = getSummaryDetails.summary_text
    AutorName = getSummaryDetails.summary_author
    word_count = summary.count(' ')
    char_count = summary.count('')
    if '.mp4' or '.avi' in getSummaryDetails.summary_org_file:
        context['fileType'] = 'video'
        context['fileURL'] = "/"+getSummaryDetails.summary_org_file

    if request.method == 'GET':

        context['userName'] = userName
        context['altercount'] = len(listofNotification)
        context['notifications'] = listofNotification
        context['summary'] = summary
        context['summary_approved'] = summary_approved
        context['available_adv_summary'] = available_adv_summary
        context['summary_id'] = summary_id
        context['available_trans'] = available_trans
        # 'adv_summary_id': adv_summary_id,
        context['AutorName'] = AutorName
        context['word_count'] = word_count
        context['char_count'] = char_count

        return render(request, 'logged_pages/viewSummary.html', context=context)


def adv_view_summary(request, summary_id):
    userName = request.session.get('UserName')
    listofNotification = listNotifications(userName)
    getAdvSummaryDetails = AdvancedSummary.objects.filter(
        adv_summary_id=summary_id)[0]

    adv_summary_text = getAdvSummaryDetails.adv_summary_text
    char_count = adv_summary_text.count('')
    word_count = adv_summary_text.count(' ')
    summary_approved = getAdvSummaryDetails.adv_summary_approval_status
    available_trans = getAdvSummaryDetails.adv_avaibale_trans
    author = getAdvSummaryDetails.adv_summary_author
    if request.method == 'GET':
        context = {
            'userName': userName,
            'altercount': len(listofNotification),
            'notifications': listofNotification,
            'AutorName': author,
            'summary_approved': summary_approved,
            'available_trans': available_trans,
            'char_count': char_count,
            'word_count': word_count,
            'summary_id': summary_id,
            'summary': adv_summary_text

        }

        return render(request, 'logged_pages/adv_view_summary.html', context=context)


def translatePage(request, type, summary_id):
    userName = request.session.get('UserName')
    uerDetails = userDet(userName)
    listofNotification = listNotifications(userName)
    context = {
        'userName': userName,
        'altercount': len(listofNotification),
        'notifications': listofNotification
    }
    if request.method == "GET":

        if type == 'summary':
            getSummaryDetails = SummaryModel.objects.filter(
                summary_id=summary_id)[0]
            SummaryTextforTrans = getSummaryDetails.summary_text
            context['summary_id'] = getSummaryDetails.summary_id
            context['summary'] = SummaryTextforTrans

            return render(request, 'logged_pages/translate.html', context=context)
        if type == 'adv_summary':
            getAdvSummaryDetails = AdvancedSummary.objects.filter(
                adv_summary_id=summary_id)[0]

            context['summary_id'] = getAdvSummaryDetails.adv_summary_id
            context['summary'] = getAdvSummaryDetails.adv_summary_text

            return render(request, 'logged_pages/translate.html', context=context)


def translatefunc(request):

    if request.method == 'POST':
        textforSummary = request.POST.get('summary_text')
        tobeConvertLang = request.POST.get('convert')
        # return HttpResponse(tobeConvertLang)
        TransTXT = translate(textforSummary, tobeConvertLang)
        # return HttpResponse(TransTXT)
        userName = request.session.get('UserName')

        usetDet = userDet(userName)
        OBJ = TranslationsModel(transLang=TransTXT.dest,
                                transText=TransTXT.text,
                                transDoneBy=usetDet,
                                )
        OBJ.save()

        # return HttpResponseRedirect('/summary/my_translates?translateid=' + str(OBJ.transID))
        # return render(request, 'logged_pages)
        return HttpResponseRedirect('/summary/viewTrans/' + str(OBJ.transID))


def my_translates(request):

    userName = request.session.get('UserName')
    usetDet = userDet(userName)
    listofNotification = listNotifications(userName)
    translist = TranslationsModel.objects.filter(transDoneBy=usetDet)[::-1]
    context = {
        'userName': userName,
        'altercount': len(listofNotification),
        'notifications': listofNotification,
        'translist': translist

    }
    return render(request, 'logged_pages/my_translates.html', context=context)


def viewTrans(request, transid):
    userName = request.session.get('UserName')
    listofNotification = listNotifications(userName)

    getTransDetails = TranslationsModel.objects.filter(transID=transid)[0]
    approval_status = getTransDetails.transApprovalStatus
    context = {
        'userName': userName,
        'altercount': len(listofNotification),
        'notifications': listofNotification,
        'trans': getTransDetails,
        'approval_status': approval_status

    }
    return render(request, 'logged_pages/viewTrans.html', context=context)


def new_content(request):
    return render(request, 'logged_pages/new_content.html')


def new_content_gen(request):
    if request.method == 'POST':
        userName = request.session.get('UserName')
        textfromForm = request.POST['text_to_gen']
        typefromForm = request.POST['type']

        if typefromForm == 'text':
            genText = parse_text_to_new_article(textfromForm)
            OBJ = ContentModal(contentType='text',
                               contentAddedBy=userDet(userName),
                               content_text=genText,
                               contentQN=textfromForm

                               )
        else:
            genImg = gen_image_from_text(textfromForm)
            OBJ = ContentModal(contentType='img',
                               contentAddedBy=userDet(userName),
                               contentIMGPath=genImg,
                               contentQN=textfromForm

                               )

        OBJ.save()

        return HttpResponseRedirect('/summary/generated_content/'+str(OBJ.contentID))


# def new_content_gen_img(request,):
#     if request.method == 'POST':
#         return HttpResponse("Img Generated")


def generated_content(request, content_id):

    fetchContentDetails = ContentModal.objects.filter(contentID=content_id)[0]
    context = {}
    if (fetchContentDetails.contentType == 'text'):
        context = {
            'contentType': 'text',
            'contentQN': fetchContentDetails.contentQN,
            'content': fetchContentDetails.content_text

        }
    else:
        context = {
            'contentType': 'img',
            'contentQN': fetchContentDetails.contentQN,
            'imgPath': fetchContentDetails.contentIMGPath

        }

    return render(request, 'logged_pages/generated_content.html', context=context)


def new_content_history(request):
    userName = request.session.get('UserName')
    userDetails = userDet(userName)
    contentHistory = ContentModal.objects.filter(
        contentAddedBy=userDetails)[::-1]

    context = {
        'contentHistory': contentHistory
    }
    return render(request, 'logged_pages/new_content_history.html', context=context)


def twitter_analysis(request):
    if request.method == 'GET':
        return render(request, 'logged_pages/twitter_analytics.html')
    else:
        userName = request.session.get('UserName')
        userDetails = userDet(userName)
        hashTagfromForm = request.POST.get('hashtag')
        Analytics = twitter_sa(hashTagfromForm, 400, userName)
        OBJ = TwitterAnalysis(

            twitter_hashtag=hashTagfromForm,
            analysis_AddedBy=userDetails,
            analysis_twitter_path=Analytics['fileName'],
            analysis_neutral=Analytics['nutral'],
            analysis_positive=Analytics['positve'],
            analysis_negative=Analytics['nagative'],
            analysis_very_nagitive=Analytics['very_negative']
        )
        OBJ.save()

        return HttpResponseRedirect('/summary/analysis_report/'+str(OBJ.analysis_id))


def twitter_analysis_reports(request):
    if request.method == 'GET':
        userName = request.session.get('UserName')
        userDetails = userDet(userName)
        analysis_reports = TwitterAnalysis.objects.filter(
            analysis_AddedBy=userDetails)[::-1]
        context = {
            'analysis_reports': analysis_reports
        }
        return render(request, 'logged_pages/twitter_analysis_reports.html', context=context)


def analysis_report(request, analysis_id):
    if request.method == 'GET':
        analalysisReport = TwitterAnalysis.objects.filter(
            analysis_id=analysis_id)[0]
        context = {
            'analalysisReport': analalysisReport,
            'analysis_positive': floor(float(analalysisReport.analysis_positive)),
            'analysis_neutral': floor(float(analalysisReport.analysis_neutral)),
            'analysis_negative': floor(float(analalysisReport.analysis_negative)),
            'analysis_very_nagitive': floor(float(analalysisReport.analysis_very_nagitive))
        }
    return render(request, 'logged_pages/twitter_analysis_report.html', context)


# SpecchDiffer


def postMeeting(request):
    if request.method == 'POST':
        userName = request.session.get('UserName')
        userDetails = userDet(userName)
        meetingFile = request.FILES.get('meetingaudio')
        numofSPK = request.POST.get('speaker_number')
        language = request.POST.get('language')
        meetingModel = request.POST.get('meetingModel')
        userDir = os.path.exists(fileDIR+"/"+userName)
        test = ""
        ts = datetime.datetime.now()
        ts = str(int(ts.strftime("%Y%m%d%H%M%S")))
        if userDir == False:
            test = os.path.join(fileDIR, userName)
            os.mkdir(test)
        else:
            test = os.path.join(fileDIR, userName)+"/" + \
                userName+"_"+ts+"_"+meetingFile.name
            # return test
        # return HttpResponse(str(test))

        handle_uploaded_file(test, meetingFile)

    returnJSON = speaker_diarization(
        test, numofSPK, language, meetingModel, userName, meetingFile.name)
    OBJ = MeetingConversionModel(
        meetingfileLocation=test, meetindAddedBy=userDetails, meetingTransscription=returnJSON.get('transcript_location'))
    OBJ.set_chunks(returnJSON.get('filepath'))
    OBJ.save()
    return HttpResponseRedirect('/summary/view_meeting_gen/'+str(OBJ.meetingID))


def mymeetings(request):
    userName = request.session.get('UserName')
    userDetails = userDet(userName)
    getListMeeting = MeetingConversionModel.objects.filter(
        meetindAddedBy=userDetails)
    context = {
        'getListMeeting': getListMeeting
    }
    return render(request, 'logged_pages/mymeetings.html', context=context)


def view_meeting_gen(request, meeting_id):
    getMeetingDetails = MeetingConversionModel.objects.filter(
        meetingID=meeting_id)[0]

    audio_chunks = getMeetingDetails.getListChunks()
    fread = open(getMeetingDetails.meetingTransscription, 'r')

    context = {
        'audio_chunks': audio_chunks,
        'fullMeetingAudio': getMeetingDetails.meetingfileLocation,
        'meeting_transcript': fread.read()
    }
    fread.close()
    return render(request, 'logged_pages/view_meeting_gen.html', context=context)


def sendApproval(request, type, id):

    OBJ = ApprovalModel(approvalType=type, approval_type_id=id)
    OBJ.save()
    type = bindType(type)
    # return HttpResponse(type)
    return HttpResponseRedirect('/summary/summary_dash?approved_id=' + str(OBJ.approvalID) + "&type=" + type)


# def sendApprovalTrans(request, transid):

#     status = "Sent for Approval"
#     dictsend = (
#         transid,
#         status
#     )
#     return HttpResponseRedirect('/summary/my_translates?transid=' + str(transid))


def send_to_twitter(request):
    text = " "

    userName = request.session.get('UserName')

    if request.method == 'GET':
        text = unquote(request.GET.get('summary_text').encode('utf-8'))
        # author = request.GET.get('author')
        tw = create_tweet(text, userName).data

        # tw_link=
        # tw = data{'id': '1600369209577738240', 'text': 'https://t.co/8q1mcI6MyK'}, includes={}, errors=[], meta={}
        return HttpResponseRedirect('/summary/summary_dash?type=twitter&tw_id='+str(tw['id'])+'&tw_link='+str(tw['text']))
        # return HttpResponse(tw)


def send_to_LinkedIn(request):
    try:
        userDet(request.session.get('UserName'))
        if request.method == 'GET':
            msg = request.GET.get('summary_text')
            link = "https://www.atomstate.com"
            link_text = "Company Web Site"
            linkedIn = post_to_linkedin(msg, link, link_text)
            return HttpResponseRedirect('/summary/summary_dash?type=linked_in')
            return linkedIn
    except Exception as e:
        return HttpResponse(e)


def listNotifications(UserName):
    ts = datetime.datetime.now()

    listofNotification = {
        'one': {'id': random.randint(1, 10000), 'timestamp': ts, 'from': 'admin', 'message': 'Need to approve this notification'},
        'two': {'id': random.randint(1, 10000), 'timestamp': ts, 'from': 'dean', 'message': 'New Account Added'},
        'three': {'id': random.randint(1, 10000), 'timestamp': ts, 'from': 'summarier', 'message': 'Not valid file uploaded'}

    }
    return listofNotification


def userDet(userName):
    userDetailsfromDB = User_Model.objects.get(user_UserName=userName)
    return userDetailsfromDB


def underDev(request):

    return render(request, 'logged_pages/underDev.html')


def bindType(type):
    if type == 'trans':
        return "Translation"
    elif type == 'summary':
        return "Summary"
    else:
        return "Advanced Summary"
# ==================
# ALGOS
# ==================


def sum_url(url):

    return True
